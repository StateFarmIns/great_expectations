from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Union

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
        # Determine alias connector. Oracle does not allow the keyword 'AS' for
        # table/subquery aliases, whereas other dialects commonly allow it.
        alias_connector = " AS "
        try:
            sa_engine = getattr(execution_engine, "engine", None)
            sa_dialect_name = sa_engine.dialect.name if sa_engine is not None else None
        except Exception:
            sa_dialect_name = None

        # Coerce the SQLAlchemy dialect name to the GXSqlDialect enum when
        # possible and use enum comparisons for dialect-specific behavior.
        gx_dialect: GXSqlDialect | None
        try:
            gx_dialect = GXSqlDialect(sa_dialect_name) if sa_dialect_name else None
        except Exception:
            gx_dialect = None

        # Oracle uses a space instead of 'AS' for table/subquery aliases.
        if gx_dialect == GXSqlDialect.ORACLE:
            alias_connector = " "

        # Determine whether the current dialect requires derived-table aliases
        # (MySQL/MariaDB and PostgreSQL require an explicit alias for derived tables).
        dialect_requires_derived_table_alias = False
        if gx_dialect in (GXSqlDialect.MYSQL, GXSqlDialect.POSTGRESQL):
            dialect_requires_derived_table_alias = True
        # MariaDB is sometimes reported as 'mariadb' by certain SQLAlchemy dialects;
        # handle that string as a fallback.
        elif isinstance(sa_dialect_name, str) and sa_dialect_name.lower() == "mariadb":
            dialect_requires_derived_table_alias = True

        if isinstance(batch_selectable, sa.Table):
            # Table objects can be formatted directly (no extra aliasing needed).
            query = query.format(batch=batch_selectable, **parameters)
        elif isinstance(batch_selectable, (sa.sql.Select, get_sqlalchemy_subquery_type())):
            # specifying a row_condition returns the active batch as a Select
            # specifying an unexpected_rows_query returns the active batch as a Subquery or Alias
            # this requires compilation & aliasing when formatting the parameterized query

            # all join queries require the user to have taken care of aliasing themselves
            if "JOIN" in query.upper():
                batch = batch_selectable.compile(compile_kwargs={"literal_binds": True})
                if dialect_requires_derived_table_alias:
                    query = query.format(
                        batch=f"({batch}){alias_connector}substituted_batch_subquery",
                        **parameters,
                    )
                else:
                    query = query.format(batch=f"({batch})", **parameters)
            else:
                aliased_batch = batch_selectable.alias("subselect")
                batch = aliased_batch.compile(compile_kwargs={"literal_binds": True})
                if dialect_requires_derived_table_alias:
                    query = query.format(
                        batch=f"({batch!s}){alias_connector}substituted_batch_subquery",
                        **parameters,
                    )
                else:
                    # Preserve previous behavior for non-MySQL-like dialects and
                    # avoid introducing an alias into the formatted string so
                    # existing unit tests continue to match expected output.
                    query = query.format(batch=f"({batch!s})", **parameters)
        else:
            # Fallback: format the batch_selectable as a parenthesized fragment.
            # Historically this did not add an explicit alias; keep that behavior
            # to match existing unit tests and avoid changing simple table
            # formatting expectations.
            query = query.format(batch=f"({batch_selectable})", **parameters)

        return query

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
