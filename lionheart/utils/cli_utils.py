from typing import Dict, List, Union
import re
import shutil
from rich_argparse import RawTextRichHelpFormatter
from lionheart.utils.global_vars import REPO_URL


# Add styles
CLI_COLORS = {
    "bg_dark": "#181818",
    "red": "#d2212d",  # "#d6000c",  # "#d73236",
    "yellow_1": "#c49700",  # "#f0a639",
    "yellow_2": "#efa12d",
    "yellow_3": "#e1972a",
    "light_yellow": "#f3b65b",
    "dark_orange": "#d9831c",
    "blue": "#1075ee",
    "green": "#4ca526",
    "light_red": "#fa5750",  # "#f08a8a",  # "#ed8282" # "#ea8080" # "#ff6b6f",
}

CLI_STYLES = {
    "bg_dark": ("bgd", "on " + CLI_COLORS["bg_dark"]),
    "color_red": ("cr", CLI_COLORS["red"]),
    "color_light_red": ("clr", CLI_COLORS["light_red"]),
    "color_yellow": ("cy", CLI_COLORS["yellow_1"]),
    "color_yellow2": ("cy2", CLI_COLORS["yellow_2"]),
    "color_yellow3": ("cy3", CLI_COLORS["yellow_3"]),
    "color_light_yellow": ("cly", CLI_COLORS["light_yellow"]),
    "color_dark_orange": ("cdo", CLI_COLORS["dark_orange"]),
    "bold": ("b", "bold"),
    "italic": ("i", "italic"),
    "underline": ("u", "underline"),
    "groups": ("h1", CLI_COLORS["yellow_1"]),
    "args": (None, CLI_COLORS["light_red"]),
}

style_tags = []
for style_name, (style_tag, style) in CLI_STYLES.items():
    RawTextRichHelpFormatter.styles[f"argparse.{style_name}"] = style
    if style_tag is not None:
        RawTextRichHelpFormatter.highlights.append(
            r"\<" + style_tag + r"\>(?P<" + style_name + r">.+?)\</" + style_tag + r"\>"
        )
        style_tags.append(style_tag)


# Custom formatter class to remove markers after formatting
class CustomRichHelpFormatter(RawTextRichHelpFormatter):
    TAGS = style_tags
    group_name_formatter = str.upper

    def _strip_html_tags(self, text):
        return CustomRichHelpFormatter.strip_html_tags(
            text=text, tags=CustomRichHelpFormatter.TAGS
        )

    def format_help(self):
        help_text = super().format_help()
        return self._strip_html_tags(help_text)

    @staticmethod
    def strip_html_tags(text: str, tags: List[str]) -> str:
        for tag_text in tags:
            text = re.sub(r"\</?" + tag_text + r"\>", "", text)
        return text

    @staticmethod
    def get_console_width():
        return shutil.get_terminal_size().columns

    @staticmethod
    def pad_to(text: str, pad_to: int, side: str = "right"):
        console_width = CustomRichHelpFormatter.get_console_width()
        if pad_to > console_width:
            pad_to = console_width
        new_text = ""
        for line in text.split("\n"):
            stripped_line = CustomRichHelpFormatter.strip_html_tags(
                text=line, tags=CustomRichHelpFormatter.TAGS
            )
            needed_padding = pad_to - len(stripped_line)
            padding = "".join([" "] * needed_padding)
            if side == "left":
                new_text += padding + line + "\n"
            else:
                new_text += line + padding + "\n"
        return new_text


LION_ASCII = """<b><cy>    :::::       </cy><cr>                </cr><cy2> =##=</cy2><cy>:.   </cy></b>
<b><cy>   -:</cy2><cly>.</cly><cy>:</cy><cly>-</cly><cy>:...    </cy><cr>               </cr><cy>       -.  </cy></b>
<b><cly>  :::</cly><cy2>- ......:                </cy2><cy>        -:  </cy></b>
<b><cly> :.  </cly><cy2>-  </cy2><cy3>..    </cy3><cy>::</cy><cr>.........</cr><cy>.....:..  ..:-.  </cy></b>
<b><cly> :-.</cly><cy2>:  </cy2><cy3>.:</cy3><cy2> . </cy2><cy>..</cy><cr>.             </cr><cy>  -</cy><cdo>.</cdo><cy>:::...    </cy></b>
<b><cy2>   .:  </cy2><cy3>.:</cy3><cy2> :</cy2><cr>:              </cr><cy>    -           </cy></b>
<b><cy2>     :.: </cy2><cy>.. </cy><cr>:           </cr><cy>      .:          </cy></b>
<b><cy2>      : </cy2><cy>:. .</cy><cr>:       ..-</cr><cy3>. :</cy3><cy>:    -          </cy></b>
<b><cy>       -   :</cy><cr>:.-.....</cr><cy3>   :  -</cy3><cy>...  :.        </cy></b>
<b><cy>       -  - </cy><cy3>- :.        :  -  </cy3><cy>.:  :       </cy></b>
<b><cy>       : :   </cy><cy3>:.:.:     .: :.   </cy3><cy>:.:        </cy></b>
<b><cy>    .-:.:       </cy><cy3>.:    :...   </cy3><cy>.::..        </cy></b>
"""

LIONHEART_ASCII = """<b>........................................</b>
<b><cy>_    _ ____ _  _</cy><cr> _  _ ____ ____ ____ ___ </cr></b>
<b><cy>|    | |  | |\ |</cy><cr> |__| |___ |__| |__/  |  </cr></b>
<b><cy>|___ | |__| | \|</cy><cr> |  | |___ |  | |  \  |  </cr></b>
                                           
<b>........................................</b>
"""

LIONHEART_STRING = "<cy>LION</cy><cr>HEART</cr>"

# TODO: Once guide_me is done, change this string.
# README_STRING = f"""See the usage guide via `lionheart guide_me` or visit the the GitHub README:\n{REPO_URL}"""
README_STRING = f"""Visit the the GitHub README:\n{REPO_URL}"""


def wrap_command_description(d):
    return f"{LIONHEART_ASCII}\n{d}\n\n{README_STRING}"


class Examples:
    def __init__(self, header="Examples:", introduction: str = "") -> None:
        self.header = header
        self.introduction = introduction
        self.examples = []

    def add_example(
        self, description: str = "", example: str = "", use_prog: bool = True
    ):
        self.examples.append((description, example, use_prog))

    def construct(self):
        string = f"<h1>{self.header.upper()}</h1>\n"
        if self.introduction:
            string += "\n" + self.introduction + "\n\n"
        for desc, ex, use_prog in self.examples:
            string += Examples.format_example(
                description=desc, example=ex, use_prog=use_prog
            )
            string += "\n\n"
        return string

    @staticmethod
    def format_example(
        description: str = "", example: str = "", use_prog: Union[bool, str] = True
    ):
        if isinstance(use_prog, str):
            prog_string = f"<b>$ {use_prog}</b> "
        else:
            prog_string = "<b>$ %(prog)s</b> " if use_prog else ""
        string = f"""---
{description}

"""
        string += (prog_string + f"{example}").replace("\n", " ")
        return string


class Guide:
    def __init__(self) -> None:
        self.elements = []

    def construct_guide(self):
        return "\n".join(self.elements)

    def add_title(self, title: str):
        self.elements.append(f"<h1>{title.upper()}</h1>")
        self.add_vertical_space()

    def add_header(self, header: str, pre_spaces=0):
        self.add_vertical_space(n=pre_spaces)
        self.elements.append(f"<b><u>{header}</u></b>")
        self.add_vertical_space()

    def add_description(self, desc: str):
        self.elements.append(desc)
        self.add_vertical_space()

    def add_example(
        self, code: str, pre_comment: str = "", use_prog: Union[bool, str] = True
    ):
        self.elements.append(
            Examples.format_example(
                description=pre_comment, example=code, use_prog=use_prog
            )
        )
        self.add_vertical_space(n=2)

    def add_vertical_space(self, n=1):
        for _ in range(n):
            self.elements.append("\u200b")


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
