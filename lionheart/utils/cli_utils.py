from typing import Dict, List, Union


def parse_thresholds(thresholds: List[str]) -> Dict[str, Union[bool, List[float]]]:
    """
    Parse the threshold names given via command line.
    """
    thresh_dict = {
        "max_j": False,
        "sensitivity": [],
        "specificity": [],
        "numerics": [],
    }

    for thresh_name in thresholds:
        if thresh_name == "max_j":
            thresh_dict["max_j"] = True
        elif thresh_name[:4] == "sens":
            try:
                thresh_dict["sensitivity"].append(float(thresh_name.split("_")[1]))
            except:  # noqa: E722
                raise ValueError(
                    f"Failed to extract sensitivity value from threshold: {thresh_name}"
                )
        elif thresh_name[:4] == "spec":
            try:
                thresh_dict["specificity"].append(float(thresh_name.split("_")[1]))
            except:  # noqa: E722
                raise ValueError(
                    f"Failed to extract specificity value from threshold: {thresh_name}"
                )
        elif thresh_name.replace(".", "").isnumeric():
            thresh_dict["numerics"].append(float(thresh_name))
        else:
            raise ValueError(f"Could not parse passed threshold: {thresh_name}")
    return thresh_dict


class EpilogExamples:
    def __init__(self, header="Examples:") -> None:
        self.header = header
        self.examples = []

    def add_example(
        self, description: str = "", example: str = "", use_prog: bool = True
    ):
        self.examples.append((description, example, use_prog))

    def construct(self):
        string = f"<h1>{self.header}</h1>\n"
        for desc, ex, use_prog in self.examples:
            prog_string = "<b>$ %(prog)s</b>" if use_prog else ""
            string += f"""---
{desc}

"""
            string += (prog_string + f"{ex}").replace("\n", " ")
            string += "\n\n"
        return string
