# Memorisation signature × llama31_8b

| text | unit | n | mean surprisal | median | frac < 0.05 | frac < 0.5 | longest run < 0.5 |
|---|---|---|---|---|---|---|---|
| Sandman (EN translation) | token | 9553 | 2.33 | 1.62 | 0.123 | 0.260 | 6 |
| Natural Stories | word | 10256 | 2.57 | 1.84 | 0.051 | 0.204 | 12 |

Sandman by thirds (mean surprisal): 2.41, 2.29, 2.30
Natural Stories longest run per story: 5, 4, 4, 5, 5, 4, 5, 8, 12, 8

Reading: a verbatim-memorised text would show a long run of near-zero surprisal (tens of
tokens) and a falling mean across thirds. Units differ between rows (token vs word), so
compare run lengths within a text family, not across. Command: see docstring.
