# Memorisation signature × qwen25_7b

| text | unit | n | mean surprisal | median | frac < 0.05 | frac < 0.5 | longest run < 0.5 |
|---|---|---|---|---|---|---|---|
| Sandman (EN translation) | token | 9553 | 2.45 | 1.72 | 0.123 | 0.255 | 6 |
| Natural Stories | word | 10256 | 2.62 | 1.84 | 0.052 | 0.207 | 8 |

Sandman by thirds (mean surprisal): 2.51, 2.41, 2.43
Natural Stories longest run per story: 5, 3, 3, 5, 5, 4, 5, 4, 7, 8

Reading: a verbatim-memorised text would show a long run of near-zero surprisal (tens of
tokens) and a falling mean across thirds. Units differ between rows (token vs word), so
compare run lengths within a text family, not across. Command: see docstring.
