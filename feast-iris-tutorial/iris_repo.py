from datetime import timedelta

from feast import BigQuerySource, Entity, Feature, FeatureView, Field, ValueType
from feast.types import Float32, Int64, String

iris_flower = Entity(name="iris_id", join_keys=["iris_id"], value_type=ValueType.INT64,)

iris_source = BigQuerySource(
    table="iris_feast_dataset.iris_feast_table",
    timestamp_field="event_timestamp",
)

iris_species_fv = FeatureView(
    name="iris_feast_table",
    entities=[iris_flower],
    ttl=timedelta(days=15),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_source,
    online=True,
    tags={"classifier_data": "iris_flower_species"},
)
