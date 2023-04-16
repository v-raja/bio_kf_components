from kfp.v2 import dsl
from kfp.v2.components.types.artifact_types import Dataset
from kfp.v2.components.types.type_annotations import Input, Output


@dsl.component(
    packages_to_install=["pandas"],
)
def offset(
        dataset: Input[Dataset],
        transformed_dataset: Output[Dataset],
        cols: str,
        offset: float,
):
    """
    Offsets the specified columns using the provided arguments.
    :param dataset: The dataset to offset.
    :param transformed_dataset: The offset dataset.
    :param cols: A comma separated list of column indices to offset. Ranges are supported. e.g. 0-2, 5. * can be used to specify all columns. e.g. *. -(cols) can be used to specify all columns except the ones specified in col_string. e.g. -0-2.
    :param offset: The offset to apply.
    """

    import pandas as pd
    from typing import List, Union

    def expand_cols(col_string: str, num_cols: int) -> Union[List[int], List[str]]:
        """Returns a list of column indices from the args string.
        col_string string is a comma separated list of column indices.
        - ranges are supported. e.g. expand_cols(0-2, 5) returns [0, 1, 2].
        - comma separated list of column indices. e.g. expand_cols('0, 1, 3-4', 5) returns [0, 1, 3, 4].
        - * can be used to specify all columns. e.g. expand_cols(*, 5) returns [0, 1, 2, 3, 4].
        - * can be used to specify all columns after a certain index. e.g. expand_cols(2-*, 5) returns [2, 3, 4].
        - * can be used to specify all columns before a certain index. e.g. expand_cols(*-2, 5) returns [0, 1, 2].
        - -(col_string) can be used to specify all columns except the ones specified in col_string.
            e.g. expand_cols(-0-2, 5) returns [3, 4].
        """
        if col_string.startswith("("):
            # Remove the outer parentheses.
            col_string = col_string[1:-1]

        if col_string == "*":
            return list(range(num_cols))

        items = list(map(lambda x: x.strip(), col_string.split(",")))
        # check if any of the items is not numeric
        if any([not str.isnumeric(item.replace("-", "").replace("*", "")) for item in items]):
            return items

        if "," in col_string:
            return list(set(sum([expand_cols(col, num_cols) for col in col_string.split(",")], [])))

        if col_string.startswith("-"):
            inner_col_string = col_string[1:]
            return list(set(range(num_cols)) - set(expand_cols(inner_col_string, num_cols)))
        elif "-" in col_string:
            start, end = col_string.split("-")
            if start == "*":
                start = 0
            else:
                start = int(start) if start else 0
            if end == "*":
                end = num_cols - 1
            else:
                end = int(end) if end else num_cols
            return list(range(start, end + 1))
        else:
            return [int(col_string)]

    df = pd.read_csv(dataset.path)
    cols = expand_cols(cols, len(df.columns))
    df[cols] = df[cols] + offset
    df.to_csv(transformed_dataset.path, index=False)


if __name__ == "__main__":
    offset.component_spec.save("component.yaml")
