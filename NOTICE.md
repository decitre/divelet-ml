# Attribution and Modification Notices

divelet-ml
Copyright 2026 Emmanuel Decitre

## Original Work

**Dive into Deep Learning** is an open-source interactive textbook developed by its original authors and contributors.

- **Website:** https://d2l.ai/
- **Source repository:** https://github.com/d2l-ai/d2l-en
- **Authors and contributors:** As credited in the original project
- **Book content license:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **License:** https://creativecommons.org/licenses/by-sa/4.0/

The initial divelet-ml compatibility assessment used the following upstream revision:

`23d7a5aecceee57d1292c56e90cce307f183bb0a`

https://github.com/d2l-ai/d2l-en/commit/23d7a5aecceee57d1292c56e90cce307f183bb0a

## Nature of the Adaptation

**divelet-ml** is an independent, browser-native adaptation of Dive into Deep Learning for JupyterLite.

The project aims to preserve the original book's educational structure, mathematical explanations, learning objectives, and exercises while adapting framework-dependent implementations for execution in browser-native Python environments.

Modifications, which vary between chapters, include:

- Replacing selected framework-specific implementations with NumPy or other browser-compatible alternatives.
- Adapting code examples for JupyterLite using Pyodide and xeus-python.
- Modifying dependency installation, dataset loading, and execution instructions where necessary.
- Introducing alternative browser-native implementations for selected machine learning tasks.
- Documenting differences in functionality, performance, and compatibility.

The adapted sections are listed in the README and published at https://decitre.github.io/divelet-ml/. Each one's notebook lists the specific changes made to it.

## Attribution of Adapted Chapters

The notebook of each adapted section opens with a one-line credit and ends with an "Attribution and changes" section that provides:

1. A reference to the corresponding chapter of the original book.
2. Identification of the upstream source revision.
3. Attribution to the original authors and contributors.
4. A description of substantive modifications.
5. Applicable license information.

Adapted educational content derived from the book is distributed under the terms of CC BY-SA 4.0.

## Source Code and Third-Party Materials

The licensing of executable code, illustrations, datasets, and other third-party materials may differ from that of the book's educational content.

Before incorporating such materials, divelet-ml will review their applicable licenses and preserve required copyright notices, attribution, and license terms.

The CC BY-SA 4.0 designation for adapted book content does not automatically apply to independently developed software or separately licensed third-party materials.

The extra notebook `02.05_extra-micrograd` adapts code from [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy (MIT License, Copyright (c) 2020 Andrej Karpathy); the license text is in that notebook's "Attribution and changes" section.

Code adapted from the book's sample code is used under its MIT-0 license. divelet-ml's own source code (the `divelet_ml` package, build scripts and tests) is licensed under the Apache License 2.0; see `LICENSE`.

## Independence

divelet-ml is not an official edition of Dive into Deep Learning.

It is not affiliated with, sponsored by, or endorsed by the book's original authors or maintainers.

divelet-ml is not affiliated with D2L Corporation or its products.

The original work and its associated rights remain with their respective authors and copyright holders.

## Project Repository

https://github.com/decitre/divelet-ml

## Third-Party Data

**Fashion-MNIST** (Xiao, Rasul and Vollgraf, 2017), used from section 4.2 onwards.

- **Source:** https://github.com/zalandoresearch/fashion-mnist
- **Copyright:** 2017 Zalando SE
- **License:** MIT, copied with the data to `data/fashion-mnist/LICENSE` on the published site

The site serves the dataset's four original gzip files unchanged, from its own host, so that notebooks fetch them without contacting another site.

### Ames Housing / Kaggle House Prices

Used in section 5.7.

Files:

- `kaggle_house_pred_train.csv`
- `kaggle_house_pred_test.csv`

- **Source:** Kaggle, "House Prices: Advanced Regression Techniques".
- **Original dataset:** Ames Housing, compiled by Dean De Cock (2011).
- **Distribution:** Dive into Deep Learning, whose host `d2l-data.s3.amazonaws.com` serves these files for the book's code (`DATA_HUB['kaggle_house_train']` and `['kaggle_house_test']`). The 5.7 notebook checks their SHA-1 hashes (`585e9cc93e70b39160e7921475f9bcd7d31219ce` and `fa19780a7b011d9b009e8bff8e99922a8ee2eb90`).
- **License and rules:** MIT, as shown in the "License" field of the competition's data page on Kaggle (captured on 17 November 2026). The page's link for it goes to a generic MIT license text, `https://www.mit.edu/~amini/LICENSE.md`, whose copyright lines name Mark Otto (2013) and Andrew Fong (2017), not the data's authors, so Kaggle gives no copyright notice specific to the data. The competition's rules, which a participant accepts, limit the data to "non-commercial purposes only, including for participating in the Competition ... and for academic research and education" (7.A) and forbid redistributing it "to any party not participating in the Competition" (7.B), so divelet-ml does not host it.
- **Reference:** De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).

The site does not host these files. Section 5.7 asks the reader to download them from the book's host and upload them to the file browser: Kaggle's competition rules (section 7.B) forbid making the data available to anyone not participating in the competition.
