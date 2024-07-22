import re
import argparse
from rich_argparse import RawTextRichHelpFormatter
from lionheart.commands import collect_samples, extract_features, predict, train_model


# Add styles
colors = {
    "red": "#d32c30",
    "yellow": "#efa12d",
    "light_yellow": "#f3b65b",
    "dark_orange": "#d9831c",
    "blue": "#1075ee",
    "green": "#4ca526",
    "light_red": "#f08a8a",  # "#ed8282" # "#ea8080" # "#ff6b6f",
}

styles = {
    "color_red": ("cr", colors["red"]),
    "color_light_red": ("clr", colors["light_red"]),
    "color_yellow": ("cy", colors["yellow"]),
    "color_light_yellow": ("cly", colors["light_yellow"]),
    "color_dark_orange": ("cdo", colors["dark_orange"]),
    "bold": ("b", "bold"),
    "italic": ("i", "italic"),
    "underline": ("u", "underline"),
    "groups": ("h1", colors["light_yellow"]),
    "args": (None, colors["light_red"]),
}

style_tags = []
for style_name, (style_tag, style) in styles.items():
    RawTextRichHelpFormatter.styles[f"argparse.{style_name}"] = style
    if style_tag is not None:
        RawTextRichHelpFormatter.highlights.append(
            r"\<" + style_tag + r"\>(?P<" + style_name + r">.+?)\</" + style_tag + r"\>"
        )
        style_tags.append(style_tag)


# Custom formatter class to remove markers after formatting
class CustomRichHelpFormatter(RawTextRichHelpFormatter):
    TAGS = style_tags

    def _strip_html_tags(self, text):
        for tag_text in CustomRichHelpFormatter.TAGS:
            # Remove bold markers
            text = re.sub(r"\</?" + tag_text + r"\>", "", text)
        return text

    def format_help(self):
        help_text = super().format_help()
        return self._strip_html_tags(help_text)

    def _format_action(self, action):
        action_help = super()._format_action(action)
        return self._strip_html_tags(action_help)

    def _format_text(self, text):
        formatted_text = super()._format_text(text)
        return self._strip_html_tags(formatted_text)


lion_ascii = """<b><cy>    :::::       </cy><cr>                </cr><cy> =##=:.   </cy></b>
<b><cy>   -:</cy><cly>.</cly><cy>:-:...    </cy><cr>               </cr><cy>       -.  </cy></b>
<b><cly>  :::</cly><cy>- ......:                </cr><cy>        -:  </cy></b>
<b><cly> :.  </cly><cy>-  ..    ::</cy><cr>.........</cr><cy>.....:..  ..:-.  </cy></b>
<b><cly> :-.</cly><cy>:  .: . ..</cy><cr>.             </cr><cy>  -</cy><cdo>.</cdo><cy>:::...    </cy></b>
<b><cy>   .:  .: :</cy><cr>:              </cr><cy>    -           </cy></b>
<b><cy>     :.: .. </cy><cr>:           </cr><cy>      .:          </cy></b>
<b><cy>      : :. .</cy><cr>:       ..-</cr><cy>. ::    -          </cy></b>
<b><cy>       -   :</cy><cr>:.-.....</cr><cy>   :  -...  :.        </cy></b>
<b><cy>       -  - - :.        :  -  .:  :       </cy></b>
<b><cy>       : :   :.:.:     .: :.   :.:        </cy></b>
<b><cy>    .-:.:       .:    :...   .::..        </cy></b>

"""

lionheart_ascii = """<b>........................................  </b>
<b><cy>_    _ ____ _  _</cy><cr> _  _ ____ ____ ____ ___ </cr></b>
<b><cy>|    | |  | |\ |</cy><cr> |__| |___ |__| |__/  |  </cr></b>
<b><cy>|___ | |__| | \|</cy><cr> |  | |___ |  | |  \  |  </cr></b>

<b>........................................  </b>
"""


def main():
    parser = argparse.ArgumentParser(
        description=f"""\n                                                                               
{lion_ascii}                                        

{lionheart_ascii}

<b>L</b>iquid B<b>i</b>opsy C<b>o</b>rrelati<b>n</b>g C<b>h</b>romatin
Acc<b>e</b>ssibility and cfDN<b>A</b> Cove<b>r</b>age
Across Cell-<b>T</b>ypes
.................

Detect Cancer from whole genome sequenced plasma cell-free DNA.

Start by <b>extracting</b> the features from a BAM file. Then <b>predict</b> whether a sample is from a cancer patient or not.

Easily <b>train</b> a new model on your own data or perform <b>cross-validation</b> to compare against the paper.
        """,
        formatter_class=CustomRichHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        # description="",
        help="additional help",
        dest="command",
    )

    # Command 1
    parser_ef = subparsers.add_parser(
        "extract_features",
        help="Extract features from a BAM file",
        description=f"{lionheart_ascii}\nEXTRACT FEATURES from a BAM file.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    extract_features.setup_parser(parser_ef)

    # Command 2
    parser_ps = subparsers.add_parser(
        "predict_sample",
        help="Predict cancer status of a sample",
        description=f"{lionheart_ascii}\PREDICT whether a sample is cancer or control.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    predict.setup_parser(parser_ps)

    # Command 3
    parser_cl = subparsers.add_parser(
        "collect",
        help="Collect predictions and/or features across samples",
        description=f"{lionheart_ascii}\COLLECT predictions and/or extracted features for multiple samples.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    collect_samples.setup_parser(parser_cl)

    # Command 4
    parser_tm = subparsers.add_parser(
        "train_model",
        help="Train a model on your own data and/or the included features",
        description=f"{lionheart_ascii}\nTRAIN A MODEL on your extracted features and/or the included features.",
        formatter_class=parser.formatter_class,
        epilog=train_model.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_tm)

    # Command 5
    parser_cv = subparsers.add_parser(
        "cross_validate",
        help="Cross-validate the cancer detection model on your own data and/or the included features",
        description=f"{lionheart_ascii}\nCROSS-VALIDATE your features with nested leave-one-dataset-out (or classic) cross-validation. "
        "Use your extracted features and/or the included features.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_cv)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
