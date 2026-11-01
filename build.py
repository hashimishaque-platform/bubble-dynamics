#!/usr/bin/env python3
"""Build the single-file page from template.html and bubble.js.

    python3 build.py                       -> index.html, a complete page for GitHub Pages
    python3 build.py --artifact OUT URL    -> page body only, with the home link pointing to URL
"""
import html
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent

# ---- search engines and link previews (full page only) ----
META = {
    'repo': 'bubble-dynamics',
    'name': 'Bubble dynamics',
    'title': 'Rayleigh-Plesset bubble dynamics simulator | Hashim Ishaque',
    'headline': 'Bubble dynamics: a single gas bubble in a sound field',
    'description': ('A gas bubble in an ultrasonic or megasonic field, as in wafer cleaning: Keller-Miksis '
                    'equation, Blake threshold and a frequency-pressure cavitation regime map.'),
    'image_alt': 'Radius of a 1 µm bubble driven at 1 MHz and 300 kPa against time, with the sound pressure below',
    'keywords': ('bubble dynamics, Rayleigh-Plesset, Keller-Miksis, acoustic cavitation, inertial cavitation, '
                 'Blake threshold, cavitation regime map, Minnaert frequency, thermal damping, water temperature, 2-propanol, megasonic cleaning, ultrasonic cleaning, wafer cleaning'),
    'languages': ['JavaScript', 'Python'],
    'about': [
        ('Rayleigh-Plesset equation', 'https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Plesset_equation'),
        ('Cavitation', 'https://en.wikipedia.org/wiki/Cavitation'),
        ('Megasonic cleaning', 'https://en.wikipedia.org/wiki/Megasonic_cleaning'),
        ('Ultrasonic cleaning', 'https://en.wikipedia.org/wiki/Ultrasonic_cleaning'),
    ],
}
SITE = 'https://hashimishaque.com'
PERSON = {
    '@type': 'Person', '@id': SITE + '/#person', 'name': 'Hashim Ishaque', 'url': SITE + '/',
    'sameAs': ['https://www.linkedin.com/in/hashimishaque/', 'https://github.com/hashimishaque-platform'],
}
WEBSITE = {'@type': 'WebSite', '@id': SITE + '/#website', 'url': SITE + '/', 'name': 'Hashim Ishaque'}
MIT = 'https://opensource.org/licenses/MIT'


def head(meta, extra=()):
    """Title, description, canonical address, icons, link preview and structured data (schema.org)."""
    url = f"{SITE}/{meta['repo']}/"
    image = url + 'social.png'
    page = {
        '@type': 'WebPage', '@id': url, 'url': url, 'name': meta['headline'], 'description': meta['description'],
        'inLanguage': 'en', 'isPartOf': {'@id': WEBSITE['@id']}, 'author': {'@id': PERSON['@id']},
        'primaryImageOfPage': {'@type': 'ImageObject', 'url': image, 'width': 1200, 'height': 630},
        'about': [{'@type': 'Thing', 'name': n, 'sameAs': u} for n, u in meta['about']],
        'keywords': meta['keywords'], 'breadcrumb': {'@id': url + '#breadcrumb'},
    }
    code = {
        '@type': 'SoftwareSourceCode', '@id': url + '#code', 'name': meta['name'], 'description': meta['description'],
        'url': url, 'codeRepository': f"https://github.com/hashimishaque-platform/{meta['repo']}",
        'programmingLanguage': meta['languages'], 'runtimePlatform': 'Web browser', 'license': MIT,
        'author': {'@id': PERSON['@id']},
    }
    crumbs = {'@type': 'BreadcrumbList', '@id': url + '#breadcrumb', 'itemListElement': [
        {'@type': 'ListItem', 'position': 1, 'name': 'Hashim Ishaque', 'item': SITE + '/'},
        {'@type': 'ListItem', 'position': 2, 'name': meta['name'], 'item': url}]}
    graph = [page, code, crumbs, WEBSITE, PERSON, *extra]
    ld = json.dumps({'@context': 'https://schema.org', '@graph': graph}, ensure_ascii=False, indent=1).replace('</', '<\\/')
    e = lambda s: html.escape(s, quote=True)
    return '\n'.join([
        f"<title>{e(meta['title'])}</title>",
        f'<meta name="description" content="{e(meta["description"])}">',
        '<meta name="author" content="Hashim Ishaque">',
        f'<link rel="canonical" href="{url}">',
        '<link rel="icon" href="/favicon.ico" sizes="32x32">',
        '<link rel="icon" href="/favicon.svg" type="image/svg+xml">',
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png">',
        '<meta property="og:type" content="website">',
        '<meta property="og:site_name" content="Hashim Ishaque">',
        f'<meta property="og:title" content="{e(meta["headline"])}">',
        f'<meta property="og:description" content="{e(meta["description"])}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{e(meta["image_alt"])}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<script type="application/ld+json">\n{ld}\n</script>',
    ])


EXTRA = ()


def build(home, full):
    src = (here / 'bubble.js').read_text(encoding='utf-8')
    # the CommonJS export block is only for Node
    code = re.sub(r"\nif \(typeof module !== 'undefined'\) \{.*?\n\}\n", "\n", src, flags=re.S)
    assert code != src, 'no export block found in bubble.js'
    page = (here / 'template.html').read_text(encoding='utf-8').replace('/*BUBBLE*/', code.strip())
    if not home.startswith('http'):
        page = page.replace('href="/*HOME*/" target="_blank" rel="noopener"', f'href="{home}"')
    page = page.replace('/*HOME*/', home)
    left = re.findall(r'/\*[A-Z]+\*/', page)
    assert not left, f'placeholders left: {left}'
    for ch in ('\u2013', '\u2014'):
        assert ch not in page, f'dash {ch!r} found'
    if full:
        page = re.sub(r'<title>.*?</title>\n', '', page, count=1)
        i = page.index('</style>') + len('</style>')
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                + head(META, EXTRA) + '\n'
                + page[:i].strip() + '\n</head>\n<body>\n' + page[i:].strip() + '\n</body>\n</html>\n')
        for ch in ('\u2013', '\u2014'):
            assert ch not in page, f'dash {ch!r} found'
    return page


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--artifact':
        out, home, full = Path(sys.argv[2]), sys.argv[3], False
    else:
        out, home, full = here / 'index.html', '/', True
    page = build(home, full)
    out.write_text(page, encoding='utf-8')
    print(f'wrote {out} ({len(page)} bytes)')
