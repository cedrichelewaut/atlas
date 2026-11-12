# DESIGN.md — design system extracted from the personal website

Source of truth: the Astro site at `~/Documents/personalwebsite`.
All citations below are relative to that root, e.g. `src/styles/site.css:56`
means `/Users/cedrichelewaut/Documents/personalwebsite/src/styles/site.css` line 56.
The single stylesheet is `src/styles/site.css` (1590 lines), inlined into every
page by `src/layouts/Base.astro:38` (`<style is:inline set:html={css}>`). There are
no other CSS files and no `<style>` blocks in components. The header comment at
`src/styles/site.css:1-11` states the intent: "Every colour is a custom property.
The page is light only."

---

## 1. Colour tokens

All colours live as custom properties in `:root` (`src/styles/site.css:52-71`).
A grep of the whole `src/` tree finds exactly six hex values, all in that block
plus the favicon; nothing is hardcoded elsewhere.

| Token | Hex | Role | Defined at |
|---|---|---|---|
| `--paper` | `#fafaf7` | Page/background (warm off-white) | `site.css:53`; also `theme-color` meta at `src/layouts/Base.astro:37`; favicon bg `public/favicon.svg:2` |
| `--ink` | `#151515` | Text, standard 1px borders, submit button fill | `site.css:54`; favicon stroke `public/favicon.svg:3` |
| `--grey` | `#151515` | "Muted" text | `site.css:55` — **same value as `--ink`**; the muted look comes only from the smaller mono size, not from a lighter colour |
| `--signal` | `#9a2400` | Red accent ("Red earth", per comment `site.css:7`) | `site.css:56`; favicon left bar `public/favicon.svg:4` |
| `--rule` | `#cfcdc6` | Decorative hairline columns on `--paper` | `site.css:57` (comment at `site.css:6`) |
| `--link` / `--link-hover` / `--link-line` | `#151515` / `#151515` / `var(--link-hover)` | Links at rest and on hover | `site.css:59-61` |

`color-scheme: light` is set at `site.css:68`; `html`/`body`/`:root` backgrounds are
`var(--paper)` and text `var(--ink)` (`site.css:69-70`, `106-110`, `112-122`).
The site is light-only and deliberately does **not** follow `prefers-color-scheme`
(comment `site.css:3-4`). A `@media (forced-colors: active)` block remaps the tokens
to system colours (`site.css:87-97`, component-level at `785-813`).

### Where `--signal` (#9a2400) IS used

- `::selection` background, text turns paper — `site.css:124-127`
- `:focus-visible` outline, `2px solid var(--signal)`, offset 2px — `site.css:179-185`
- `.sheet` left edge, `border-left: 14px solid var(--signal)` — `site.css:213-221`
- `.hl` highlight background (intro name + intro lines on the home page, used in `src/pages/index.astro:88,93-97`), text paper — `site.css:414-420`
- `.pill` hover/focus background — `site.css:444-449`
- `.section-no`, `.digest-title .num`, `.poster-no`, `.flag` text colour — `site.css:318-327`, `519-526`, `556-559`, `1082-1085`
- Thick 8px rules: `border-bottom` on `.digest-title` and `.part-title`, `border-top` on `.poster` and `.shelf` — `site.css:512`, `768`, `534`, `627`
- `.lang` language badge border, `1px solid var(--signal)` — `site.css:573-591`
- `.now` block: full `background: var(--signal)` with paper text — `site.css:706-711`
- Form underlines on hover/focus: `border-bottom-color: var(--signal)` — `site.css:1153-1158`, `1523-1526`
- `.submit:hover` background + border — `site.css:1188-1192`
- `.circle:hover` background + border — `site.css:1402-1408`
- Favicon left bar — `public/favicon.svg:4`

### Where red is NOT used

- Links: rest **and** hover stay `--ink` (#151515) — `site.css:59-61`. Note: the header comment at `site.css:8-9` says hover turns links red, but the operative token is `--link-hover: #151515` (`site.css:61`) with the inline comment "Hover and focus stay ink; a 1px underline is the only change" (`site.css:58`). The values win: **links never turn red**.
- All standard structural borders (header, intro, meta, table rows, footer, frames, images) are `1px solid var(--ink)` — e.g. `site.css:315`, `344`, `968-969`, `1062`, `1200`, `1460`, `1025`.
- Decorative hairlines use `--rule` (#cfcdc6), never red — `site.css:224-242`.
- Body text and "muted" meta text are `--ink`/`--grey`, both #151515 — `site.css:54-55`.
- Form underlines at rest are `--ink`; red appears only on hover/focus — `site.css:1142`, `1516`.

---

## 2. Typography

### Fonts and loading

Self-hosted `.woff2` via five `@font-face` blocks at the top of the stylesheet,
all with `font-display: swap` (`site.css:12-50`). No CDN, no Google Fonts.
Files live in `public/fonts/` and are served with `Cache-Control: public,
max-age=31536000, immutable` (`netlify.toml`, `/fonts/*` header). Licences:
`public/fonts/AILERON-CC0.txt`, `public/fonts/IBM-PLEX-MONO-OFL.txt`.

| Family | Weights/styles loaded | Files | `@font-face` |
|---|---|---|---|
| Aileron | 400, 700 (normal) | `aileron-400.woff2`, `aileron-700.woff2` | `site.css:12-18`, `20-26` |
| IBM Plex Mono | 400, 700, 400 italic | `IBMPlexMono-Regular.woff2`, `IBMPlexMono-Bold.woff2`, `IBMPlexMono-Italic.woff2` | `site.css:28-34`, `36-42`, `44-50` |

Fallback stacks: `"Aileron", Helvetica, Arial, sans-serif` for display/body
(`site.css:116`, `150`); `"IBM Plex Mono", ui-monospace, monospace` for meta/nav
(`site.css:200`, `283`). `font-synthesis: none` on body (`site.css:120`), so no
faux bold/italic.

Usage: Aileron for headings and body; IBM Plex Mono for all metadata, nav, labels,
tables' captions/headers, article prose, buttons, and form labels
(`.meta-face` group `site.css:273-287`; `.prose p, .prose li` `site.css:475-481`;
`.lede` `site.css:483-488`).

### Type scale

Root size: `html { font-size: 18px }` (`site.css:106-109`), so `1rem = 18px`.
Only three size tokens exist (`site.css:65-67`):

| Token | Value | Derived px | Line-height | Used for |
|---|---|---|---|---|
| `--body` | `1rem` | 18px | 1.45 (`site.css:119`) | body text, dd, table cells |
| `--meta` | `0.72rem` | ≈12.96px | 1.4 (1.5 in prose lists, 1.3 `.lang`) | all mono metadata (`site.css:284-286`, `480`, `585`) |
| `--heading` | `1.15rem` | ≈20.7px | 1.35 | h1–h4, section titles (`site.css:152-153`) |

- All headings `h1–h4` are the **same size and weight 400**, `letter-spacing: 0`
  (`site.css:149-161`) — there is no display size; `h1` equals `--heading` (`site.css:157-161`).
- Headings stay in normal case (no `text-transform` on headings; comment `site.css:2`).
  The only `text-transform: uppercase` is the `.lang` badge (`site.css:587`) with
  `letter-spacing: 0.08em` (`site.css:584`); `.mark` uses `0.06em` (`site.css:1099`).
- `text-wrap: balance` on `.poster-title` (`site.css:570`). `text-rendering:
  optimizeLegibility` on body (`site.css:121`).
- Measure (line length): 66ch for copy blocks (`site.css:401`, `471`, `493`, `845`,
  `908`, `959`), 80ch for `.prose p/li` (`site.css:477`), 70ch for `.prose`
  (`site.css:997`), 36ch for `.now-title` (`site.css:723`).

---

## 3. Layout, spacing, borders, radius, shadows

### Layout

- Page shell `.sheet`: `max-width: var(--max)` (= **1440px**, `site.css:64`),
  `margin: 0`, `padding: var(--gutter)` — `site.css:213-221`.
- Grid `.grid`: `repeat(var(--cols), minmax(0, 1fr))` with `column-gap: var(--gutter)`;
  `--cols: 12` → 8 at ≤1100px → 4 at ≤720px — `site.css:263-266`, `62-63`, `73-85`.
- Two decorative 1px vertical hairlines in `--rule` at `calc(var(--gutter) * 0.38)`
  from each edge of `.sheet`, via `::before`/`::after` — `site.css:224-242`.
- Breakpoints: **1100px** and **720px** only (`site.css:73`, `80`; used throughout).
- Intro: grid `minmax(0,1fr) minmax(0,3fr)`, areas `photo text`, stacks at ≤720px —
  `site.css:337-346`, `371-384`.

### Spacing scale

No numeric 4px-step scale is defined ("not defined"). Spacing is driven by one
token plus ad-hoc rem margins:

- `--gutter: 32px` desktop, `24px` ≤1100px, `16px` ≤720px (`site.css:62`, `74`, `81`).
- Common rem margins in use: `0.35rem`, `0.45rem`, `0.55rem`, `0.7rem`, `0.75rem`,
  `0.85rem`, `1.1rem`, `1.15rem`, `1.35rem`, and `calc(var(--gutter) * 1.5)` /
  `calc(var(--gutter) * 2)` for section gaps (`site.css:501`, `763`, `776`).

### Borders and radius

- Standard rule: `1px solid var(--ink)` (header/intro/meta/table/footer/frames/images).
- Thick accent rule: `8px solid var(--signal)` (see §1).
- Sheet edge: `border-top`/`border-bottom: 1px solid var(--ink)` + `border-left: 14px
  solid var(--signal)` (`site.css:218-220`).
- Hairlines: `1px var(--rule)` (`site.css:231-232`).
- Tables: `border-collapse: collapse` (`site.css:1044`); row `border-bottom: 1px
  solid var(--ink)`; cell separators `border-left: 1px solid var(--ink)` (`site.css:1062`, `1072`).
- **Border radius:** square corners everywhere. Explicit `border-radius: 0` on
  inputs/textarea (`site.css:1143`), `.submit` (`site.css:1182`), `.unlock input`
  (`site.css:1517`); the only radius is `border-radius: 50%` on `.circle` icon
  buttons (`site.css:1386`). Everything else is default 0 ("not defined" elsewhere).

### Shadows

**None.** No `box-shadow`/`filter: drop-shadow` anywhere except one functional hack:
the `-webkit-autofill` override paints the field background with
`box-shadow: 0 0 0 1000px var(--paper) inset` (`site.css:1160-1164`) — not a shadow visually.

---

## 4. Links, buttons, inputs, states, motion

### Links

- Rest: `color: var(--link)` (#151515), `font-weight: 400`, `text-decoration: none` — `site.css:163-168`.
- Hover/focus: colour stays `--link-hover` (#151515); adds `text-decoration: underline`,
  `1px` thickness, `text-underline-offset: 0.16em` — `site.css:170-177`.
  Contextual offsets elsewhere: `0.15em` (`site.css:992`, `1017`, `1212`, etc.), `0.08em` on shelf titles (`site.css:659`).
- In the red `.now` block, links are paper and always underlined — `site.css:749-756`.
- Print: all links forced underlined — `site.css:1322-1323`.

### Focus (keyboard)

`a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible`
→ `outline: 2px solid var(--signal); outline-offset: 2px` (`site.css:179-185`).
Exceptions: `.pill:focus-visible` uses `outline-color: var(--ink)` (`site.css:451-453`);
inside `.now`, `outline-color: var(--paper)` (`site.css:758-760`). Skip link `.skip`
reveals on focus (`site.css:187-211`).

### Buttons

- Base `button` reset: `font: inherit; color: inherit; background: none; border: 0;
  padding: 0; cursor: pointer` + 100ms transition — `site.css:1166-1174`.
- `.submit` (form send): solid — `background: var(--ink); color: var(--paper);
  border: 1px solid var(--ink); border-radius: 0; padding: 0.5rem 0.75rem` in IBM Plex
  Mono `--meta` — `site.css:1176-1187`. Hover: `background: var(--signal);
  border-color: var(--signal); color: var(--paper)` — `site.css:1188-1192`.
- `.circle` (carousel arrows, social icons): `2.6rem × 2.6rem`, `border: 1px solid
  var(--ink)`, `border-radius: 50%`, transparent bg, ink icon (SVG `stroke:
  currentColor`, `stroke-width: 1.6`) — `site.css:1378-1400`. Hover/focus-visible:
  `background: var(--signal); border-color: var(--signal); color: var(--paper)` —
  `site.css:1402-1408`. Disabled: `opacity: 0.3; cursor: default`, hover reverted —
  `site.css:1410-1419`.
- `.pill` (in-text link chip, e.g. "Get in touch →"): ink text at rest; on
  hover/focus-visible gets `background: var(--signal); color: var(--paper)` —
  `site.css:435-453`.

### Inputs

- `.field input/textarea`: `background: transparent`, `border: 0` except
  `border-bottom: 1px solid var(--ink)`, `border-radius: 0`, `padding: 0.35rem 0`,
  Aileron `--body`, ink text — `site.css:1131-1147`. Textarea `min-height: 4.5rem;
  resize: vertical` (`site.css:1148-1151`).
- Hover **and** focus: `border-bottom-color: var(--signal)` — `site.css:1153-1158`.
  Plus the global `:focus-visible` 2px signal outline.
- Labels: IBM Plex Mono `--meta`, `color: var(--grey)` — `site.css:1124-1129`.
- `.unlock` password field repeats the same underline style (`site.css:1512-1526`).
- Form width: `.contact-form { max-width: 34rem }` (`site.css:1431-1434`).

### Hover/focus summary

| Element | Rest | Hover / focus |
|---|---|---|
| `a` | ink, no underline | ink + 1px underline, offset 0.16em (`site.css:170-177`) |
| `.pill` | ink text | signal bg, paper text (`site.css:444-449`) |
| `.submit` | ink bg, paper text | signal bg/border, paper text (`site.css:1188-1192`) |
| `.circle` | transparent, ink border/icon | signal bg/border, paper icon (`site.css:1402-1408`) |
| inputs | ink bottom border | signal bottom border (`site.css:1153-1158`) |
| `::selection` | — | signal bg, paper text (`site.css:124-127`) |
| everything focus-visible | — | 2px solid signal outline, offset 2px (`site.css:179-185`) |

### Motion

- Only `transition: ... 100ms linear` — on `a`: color, text-decoration-color,
  background-color, border-color (`site.css:167`); on `button`: color,
  background-color, border-color (`site.css:1173`); on inputs: border-color
  (`site.css:1145`). Duration 100ms, easing linear, no delay.
- No `@keyframes`, no `animation` anywhere (grep-verified: "not defined").
- `prefers-reduced-motion`: **not defined** — no media query exists.
- Carousel uses `scroll-snap-type: x mandatory` and hides its scrollbar
  (`scrollbar-width: none`) — `site.css:1326-1339`, not animation.

### Other states

- Loading/empty/error/clarification component states: **not defined** (no such components in CSS).
- Print stylesheet `@media print`: hides skip link + hairlines, removes sheet
  borders/max-width, underlines all links — `site.css:1308-1324`.

---

## 5. Ready-to-paste `:root` block

Verbatim from `src/styles/site.css:52-71`, with the breakpoint overrides that also
target `:root` (`site.css:73-85`) and the forced-colors remap (`site.css:87-97`):

```css
:root {
  --paper: #fafaf7;
  --ink: #151515;
  --grey: #151515;
  --signal: #9a2400;
  --rule: #cfcdc6;
  --link: #151515;
  --link-line: var(--link-hover);
  --link-hover: #151515;
  --gutter: 32px;
  --cols: 12;
  --max: 1440px;
  --body: 1rem;
  --meta: 0.72rem;
  --heading: 1.15rem;
  color-scheme: light;
  background: var(--paper);
  color: var(--ink);
}

@media (max-width: 1100px) {
  :root {
    --gutter: 24px;
    --cols: 8;
  }
}

@media (max-width: 720px) {
  :root {
    --gutter: 16px;
    --cols: 4;
  }
}

@media (forced-colors: active) {
  :root {
    --paper: Canvas;
    --ink: CanvasText;
    --grey: CanvasText;
    --signal: LinkText;
    --rule: GrayText;
    --link: LinkText;
    --link-line: LinkText;
    --link-hover: CanvasText;
  }
}
```

Companion base rules worth pasting with it (`site.css:106-122`):

```css
html { background: var(--paper); color: var(--ink); font-size: 18px; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Aileron", Helvetica, Arial, sans-serif;
  font-size: var(--body);
  font-weight: 400;
  line-height: 1.45;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}
```

---

## 6. Do / Don't — the minimal style in one list

**Do**

- Use only the six tokens: paper `#fafaf7`, ink `#151515`, grey `#151515`, signal
  `#9a2400`, rule `#cfcdc6`, link `#151515`. Every colour is a custom property
  (`site.css:1-11`) — never hardcode a hex in a component.
- Light only: `color-scheme: light`, never follow `prefers-color-scheme`
  (`site.css:3-4`, `68`); remap tokens under `forced-colors: active` (`site.css:87-97`).
- Reserve red `--signal` for: the 14px left edge of the sheet, 8px thick rules under
  section titles, the intro highlight `.hl`, the `.now` block, section numbers
  (`.section-no`, `.num`, `.poster-no`, `.flag`), the `.lang` badge border, hover
  fills on `.pill`/`.submit`/`.circle`, input underlines on hover/focus, the focus
  outline, and `::selection` (§1 list with lines).
- Keep links ink at rest and on hover; the 1px underline is the only hover change
  (`site.css:58-61`, `163-177`).
- All structural borders `1px solid var(--ink)`; the two vertical hairlines are the
  only `--rule` elements and sit at `calc(var(--gutter) * 0.38)` (`site.css:224-242`).
- Square corners everywhere (`border-radius: 0`); the only curve is `50%` on circle
  icon buttons (`site.css:1386`). No shadows (`site.css` has none).
- Type in two families only: Aileron 400/700 for text and headings, IBM Plex Mono
  400/700/italic for all metadata; self-host woff2 with `font-display: swap`
  (`site.css:12-50`).
- Three sizes only: `--body` 1rem, `--meta` 0.72rem, `--heading` 1.15rem on an 18px
  root; headings all weight 400, same size as h1, `letter-spacing: 0` (`site.css:109`, `149-161`).
- Space with `--gutter` (32/24/16px) and small rem margins; layout on a 12/8/4-column
  grid inside a 1440px sheet (`site.css:62-66`, `213-221`, `263-266`).
- Motion: 100ms linear transitions on colour/background/border only; nothing else
  (`site.css:167`, `1145`, `1173`).

**Don't**

- Don't introduce a second accent, a dark theme, gradients, or any colour outside the
  six tokens (`site.css:1-11`, `68`).
- Don't turn links red on hover — hover is ink + underline (`site.css:61`, `170-177`);
  ignore the stale comment at `site.css:8-9` that says otherwise.
- Don't use red for body text, table rules, hairlines, or resting form borders —
  those are ink/rule (`site.css:57`, `1062`, `1142`).
- Don't add border radius (except `.circle` 50%), drop shadows, keyframe animations,
  or a `prefers-reduced-motion` block — none exist in the source.
- Don't add heading sizes/weights beyond 400 at `--heading`, and don't uppercase
  headings (only `.lang` is uppercased) (`site.css:149-161`, `587`).
- Don't set a "muted" lighter grey: `--grey` is deliberately identical to `--ink`
  (`site.css:54-55`); muted means smaller mono type, not a lighter colour.
- Don't exceed the 1440px sheet, the 1100px/720px breakpoints, or the 66/70/80ch
  measures (`site.css:64`, `73`, `80`, `401`, `477`, `997`).

---

## Appendix: files read

- `src/styles/site.css` (1590 lines — the entire design system)
- `src/layouts/Base.astro` (theme-color meta line 37; CSS inlined line 38)
- `src/layouts/Note.astro`, `src/pages/index.astro` (`.hl`/`.pill` usage)
- `src/site.config.ts`, `src/site.ts`, `src/site.sections.ts`, `astro.config.ts`,
  `netlify.toml` (config; fonts cache header)
- `public/favicon.svg`, `public/fonts/` (font files + licences)
