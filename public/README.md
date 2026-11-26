# Public writing

Learning in public is part of From Thin Air. Posts about the project live in this folder, next to the code, together with their images and their sources. The [journal](../journal/index.html) is the record of the work, and the posts draw on it.

## Posts

| # | Post | Platform | Status |
| --- | --- | --- | --- |
| 0001 | PLAD, PLLM, and From Thin Air | LinkedIn | Draft, not published. The text is in [`0001-plad-pllm/post.md`](0001-plad-pllm/post.md) and the renders are in [`0001-plad-pllm/images/`](0001-plad-pllm/images). A shorter plain-text version for the LinkedIn feed is in [`0001-plad-pllm/linkedin.txt`](0001-plad-pllm/linkedin.txt). It has no `sources.md` yet. |

## Layout

Each post gets one folder, named with a four-digit number and a short slug, such as `0001-plad-pllm/`.

| Path | What it holds |
| --- | --- |
| `README.md` | This page: the list of posts, each with its platform and status. |
| `<NNNN-slug>/post.md` | The article: the long version, with headings and images. |
| `<NNNN-slug>/linkedin.txt` | The feed post: the short plain-text version, ready to paste into LinkedIn. |
| `<NNNN-slug>/images/` | The rendered images the post uses. |
| `<NNNN-slug>/make_images.py` | The script that makes those images from data in the repo. Running it rebuilds every image. |
| `<NNNN-slug>/sources.md` | The papers and links the post cites. |

## Rules

- Every post points to the journal entries it draws on, and every number in a post comes from a journal entry or a script in the repo.
- An image that shows a result is rendered from real traces. An image that only explains an idea is labeled as an illustration.
- A post about a result waits until the result exists. A post about what was learned can go out at any time.
- Authors are credited by name, and every paper is linked.
- Figures are our own renders. Figures from other papers are not copied here.
