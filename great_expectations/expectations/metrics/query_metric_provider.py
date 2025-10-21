from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Union, cast

from typing_extensions import NotRequired, TypedDict

from great_expectations.compatibility.sqlalchemy import (
    sqlalchemy as sa,
)
from great_expectations.execution_engine.sqlalchemy_dialect import GXSqlDialect
from great_expectations.expectations.metrics.metric_provider import MetricProvider
from great_expectations.expectations.metrics.util import MAX_RESULT_RECORDS
from great_expectations.util import get_sqlalchemy_subquery_type

if TYPE_CHECKING:
    from great_expectations.execution_engine import SqlAlchemyExecutionEngine

logger = logging.getLogger(__name__)


class MissingElementError(TypeError):
    def __init__(self):
        super().__init__(
            "The batch subquery selectable does not contain an "
            "element from which query parameters can be extracted."
        )


class MissingParameterError(ValueError):
    def __init__(self, parameter_name: str):
        super().__init__(f"Must provide `{parameter_name}` to `{self.__class__.__name__}` metric.")


class InvalidParameterTypeError(TypeError):
    def __init__(self, parameter_name: str, expected_type: type):
        super().__init__(f"`{parameter_name}` must be provided as type `{expected_type}`.")


class QueryParameters(TypedDict):
    column: NotRequired[str]
    column_A: NotRequired[str]
    column_B: NotRequired[str]
    columns: NotRequired[list[str]]


class QueryMetricProvider(MetricProvider):
    """Base class for all Query Metrics, which define metrics to construct SQL queries.

     An example of this is `query.table`,
     which takes in a SQL query & target table name, and returns the result of that query.

     In some cases, subclasses of MetricProvider, such as QueryMetricProvider, will already
     have correct values that may simply be inherited by Metric classes.

     ---Documentation---
         - https://docs.greatexpectations.io/docs/guides/expectations/creating_custom_expectations/how_to_create_custom_query_expectations

    Args:
         metric_name (str): A name identifying the metric. Metric Name must be globally unique in
             a great_expectations installation.
         domain_keys (tuple): A tuple of the keys used to determine the domain of the metric.
         value_keys (tuple): A tuple of the keys used to determine the value of the metric.
         query (str): A valid SQL query.
    """

    domain_keys = ("batch_id", "row_condition", "condition_parser")

    query_param_name: ClassVar[str] = "query"

    dialect_columns_require_subquery_aliases: ClassVar[set[GXSqlDialect]] = {
        GXSqlDialect.POSTGRESQL
    }

    @classmethod
    def _get_query_from_metric_value_kwargs(cls, metric_value_kwargs: dict) -> str:
        query_param = cls.query_param_name
        query: str | None = metric_value_kwargs.get(query_param) or cls.default_kwarg_values.get(
            query_param
        )
        if not query:
            raise MissingParameterError(query_param)
        if not isinstance(query, str):
            raise InvalidParameterTypeError(query_param, str)

        return query

    @classmethod
    def _get_parameters_dict_from_query_parameters(
        cls, query_parameters: Optional[QueryParameters]
    ) -> dict[str, Any]:
        if not query_parameters:
            return {}
        elif query_parameters and "columns" in query_parameters:
            columns = query_parameters.pop("columns")
            query_columns = {f"col_{i}": col for i, col in enumerate(columns, 1)}
            return {**query_parameters, **query_columns}
        else:
            return {**query_parameters}

    @classmethod
    def _get_substituted_batch_subquery_from_query_and_batch_selectable(
        cls,
        query: str,
        batch_selectable: sa.Selectable,
        execution_engine: SqlAlchemyExecutionEngine,
        query_parameters: Optional[QueryParameters] = None,
    ) -> str:
        parameters = cls._get_parameters_dict_from_query_parameters(query_parameters)
        user_provided_alias = cls._detect_user_provided_alias(query)

        if isinstance(batch_selectable, sa.Table):
            # Table objects can be formatted directly (no extra aliasing needed).
            return query.format(batch=batch_selectable, **parameters)

        if isinstance(batch_selectable, (sa.sql.Select, get_sqlalchemy_subquery_type())):
            batch_fragment = cls._format_batch_selectable_for_query(
                batch_selectable=batch_selectable,
                query=query,
                execution_engine=execution_engine,
                user_provided_alias=user_provided_alias,
            )
            return query.format(batch=batch_fragment, **parameters)

        return query.format(batch=f"({batch_selectable})", **parameters)

    @classmethod
    def _detect_user_provided_alias(cls, query: str) -> bool:
        """Detect whether the user provided an alias after `{batch}`.

        Returns True only when a valid identifier follows `{batch}` (optionally
        preceded by `AS`) and that identifier is not a common SQL keyword.
        """
        alias_match = re.search(
            r"\{batch\}\s*(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
            query,
            flags=re.IGNORECASE,
        )
        if not alias_match:
            return False
        candidate = alias_match.group(1)
        sql_keywords = {
            "WHERE",
            "JOIN",
            "ON",
            "USING",
            "GROUP",
            "ORDER",
            "LIMIT",
            "HAVING",
            "UNION",
            "EXCEPT",
            "INTERSECT",
        }
        return candidate.upper() not in sql_keywords

    @classmethod
    def _format_batch_selectable_for_query(
        cls,
        batch_selectable: sa.Selectable,
        query: str,
        execution_engine: SqlAlchemyExecutionEngine,
        user_provided_alias: bool,
    ) -> str:
        """Compile and format a Select/Subquery for insertion into the query.

        This centralizes the compile/aliasing rules so the main method stays
        small and easier to lint.
        """
        alias_connector, dialect_requires_derived_table_alias = (
            cls._get_aliasing_behavior_for_engine(execution_engine)
        )

        # If the query contains a JOIN, callers are expected to provide their
        # own aliasing; compile the raw selectable and append our alias only
        # when the dialect requires it and the user did not provide one.
        if "JOIN" in query.upper():
            compiled = cast("Any", batch_selectable).compile(compile_kwargs={"literal_binds": True})
            if dialect_requires_derived_table_alias and not user_provided_alias:
                return f"({compiled}){alias_connector}substituted_batch_subquery"
            return f"({compiled})"

        # If the caller already supplied an alias token after `{batch}`, compile
        # the raw selectable and preserve the user's alias usage.
        if user_provided_alias:
            compiled = cast("Any", batch_selectable).compile(compile_kwargs={"literal_binds": True})
            return f"({compiled})"

        # Default: compile an aliased selectable. Append derived-table alias
        # only for dialects that require it; otherwise, embed compiled SQL
        # without adding an substituted alias to preserve historical behavior.
        aliased_batch = cast("Any", batch_selectable).alias("subselect")
        compiled = cast("Any", aliased_batch).compile(compile_kwargs={"literal_binds": True})
        if dialect_requires_derived_table_alias and not user_provided_alias:
            return f"({compiled!s}){alias_connector}substituted_batch_subquery"
        return f"({compiled!s})"

    @classmethod
    def _get_sqlalchemy_records_from_substituted_batch_subquery(
        cls,
        substituted_batch_subquery: str,
        execution_engine: SqlAlchemyExecutionEngine,
    ) -> list[dict]:
        result: Union[Sequence[sa.Row[Any]], Any] = execution_engine.execute_query(
            sa.text(substituted_batch_subquery)
        ).fetchmany(MAX_RESULT_RECORDS)

        if isinstance(result, Sequence):
            return [element._asdict() for element in result]
        else:
            return [result]

    @classmethod
    def _get_aliasing_behavior_for_engine(
        cls, execution_engine: SqlAlchemyExecutionEngine
    ) -> tuple[str, bool]:
        """Return (alias_connector, dialect_requires_derived_table_alias).

        Using a helper reduces complexity in the main substitution method and
        consolidates dialect mapping logic in one place.
        """
        alias_connector = " AS "
        try:
            sa_engine = getattr(execution_engine, "engine", None)
            sa_dialect_name = sa_engine.dialect.name if sa_engine is not None else None
        except Exception:
            sa_dialect_name = None

        # Normalize dialect name (strip driver suffix like '+pymysql') and
        # try to coerce to GXSqlDialect enum; fall back to string checks.
        dialect_name_base = None
        if isinstance(sa_dialect_name, str):
            dialect_name_base = sa_dialect_name.split("+", 1)[0]

        gx_dialect: GXSqlDialect | None
        try:
            gx_dialect = GXSqlDialect(dialect_name_base) if dialect_name_base else None
        except Exception:
            gx_dialect = None

        if gx_dialect is GXSqlDialect.ORACLE:
            alias_connector = " "

        dialect_requires_derived_table_alias = False
        if (
            isinstance(gx_dialect, GXSqlDialect)
            and gx_dialect in (GXSqlDialect.MYSQL, GXSqlDialect.POSTGRESQL)
        ) or (isinstance(dialect_name_base, str) and dialect_name_base.lower() == "mariadb"):
            dialect_requires_derived_table_alias = True

        return alias_connector, dialect_requires_derived_table_alias
