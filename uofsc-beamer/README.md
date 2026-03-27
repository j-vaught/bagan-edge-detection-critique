# University of South Carolina — LaTeX Beamer Template (16:9)

A clean Beamer presentation template using the **official UofSC brand colors** (Garnet `#73000A`, Black, and accent palette) and a public-safe font stack that mirrors the current UofSC typography guidance.

| Role | Official UofSC Font | Bundled Public Fallback | Common Local Substitute |
|------|----------------------|-------------------------|-------------------------|
| **Titles** (ALL CAPS) | Berlingske Sans Extra-Condensed Extra-Bold | Oswald Bold | Impact |
| **Formal / serif** | Berlingske Serif | Tinos | Georgia |
| **Body / everything else** | Berlingske Sans | Arimo | Arial |

The public repository intentionally does **not** bundle the licensed Berlingske brand fonts or proprietary system fonts. If the Berlingske fonts are already installed on your machine, `beamerthemeuofsc.sty` will prefer them automatically. Otherwise it uses the bundled redistributable fallback fonts in `fonts/`.

## Files

```
├── main.tex                 ← edit this
├── beamerthemeuofsc.sty     ← theme (colors, fonts, layout)
├── fonts/
│   ├── Oswald-Bold.ttf      ← title font (≈ Impact)
│   ├── Oswald-Regular.ttf
│   ├── Arimo-Regular.ttf    ← body font (≈ Arial)
│   ├── Arimo-Bold.ttf
│   ├── Arimo-Italic.ttf
│   ├── Arimo-BoldItalic.ttf
│   ├── Tinos-Regular.ttf    ← formal font (≈ Georgia)
│   ├── Tinos-Bold.ttf
│   ├── Tinos-Italic.ttf
│   └── Tinos-BoldItalic.ttf
└── README.md
```

## Quick Start

**Requires XeLaTeX** (for custom font support via `fontspec`).

```bash
xelatex main.tex
xelatex main.tex   # run twice for TOC
```

Or with `latexmk`:

```bash
latexmk -xelatex main.tex
```

## Official Font Support

The theme checks for these local font names first:

- `Berlingske Sans Extra-Condensed Extra-Bold`
- `Berlingske Sans`
- `Berlingske Serif`

If they are not installed, the template stays on the bundled fallback stack:

- `Oswald` for display titles
- `Arimo` for body sans-serif text
- `Tinos` for formal serif text

## Title Uppercase Behavior

All frame titles and section dividers render in **Oswald Bold, ALL CAPS** by default — matching UofSC's Impact-style convention.

To use **mixed case** on a specific slide, wrap the title:

```latex
\begin{frame}{\nouppercase{This Stays Mixed Case}}
  ...
\end{frame}
```

## Using the Georgia (Tinos) Font

The body default is Arimo (Arial). To switch a block of text to the formal serif font:

```latex
\formal{This text appears in Tinos (Georgia-style).}
```

Or use `\georgiafont` directly for longer passages.

## Adding the UofSC Logo

1. Download a **white** version of the logo from the [Brand Toolbox](https://sc.edu/about/offices_and_divisions/communications/toolbox/visuals/logos/).
2. Place it in the project root (e.g., `uofsc-logo-white.png`).
3. Uncomment this line in `main.tex`:

```latex
\titlegraphic{\includegraphics[height=1.2cm]{uofsc-logo-white.png}}
```

## Theme Features

- **16:9 widescreen** (change `aspectratio=169` to `aspectratio=43` for 4:3)
- Garnet frame title bar with dark garnet accent line
- Garnet footer with author · short title · slide number
- Full-garnet section divider slides (auto-generated)
- Full-garnet title and closing slides
- Three block styles: standard (garnet), alert (rose), example (congaree)
- All official UofSC colors available as LaTeX color names

## Available Color Names

**Primary:** `UofSCGarnet`, `UofSCBlack`, `UofSCWhite`
**Neutral:** `UofSCGray90`, `UofSCGray70`, `UofSCGray50`, `UofSCGray30`, `UofSCGray10`, `UofSCWarmGrey`, `UofSCSandstorm`
**Accent:** `UofSCRose`, `UofSCAtlantic`, `UofSCCongaree`, `UofSCHorseshoe`, `UofSCGrass`, `UofSCHoneycomb`
**Special:** `UofSCDarkGarnet`, `UofSCAzalea`

## License

The template source in this repository is released under the MIT License. Bundled font files stay under their upstream licenses:

- `Oswald` under SIL Open Font License 1.1
- `Arimo` under Apache License 2.0
- `Tinos` under Apache License 2.0

See `fonts/README.md` and the license files in `fonts/` for the exact notices. UofSC names, logos, and licensed brand font files are not included in the MIT grant.
