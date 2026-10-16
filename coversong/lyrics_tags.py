"""Song text with singer tags (as on Genius): "[Verse 1: Karina]", "[Chorus: Winter, Giselle]", "[Куплет 2: Name]" (Russian tags too).

parse() reads who sings each section; align() puts that onto Whisper's words of the lead vocal, so the
multi-voice tab can tell who sings when without guessing by timbre.
"""
import difflib
import re
import unicodedata

SECTIONS = ("verse", "chorus", "pre-chorus", "prechorus", "post-chorus", "postchorus", "bridge", "intro", "outro",
            "hook", "refrain", "interlude", "breakdown", "rap", "drop", "part", "instrumental", "skit", "spoken",
            "куплет", "припев", "предприпев", "пред-припев", "бридж", "вступление", "интро", "аутро", "проигрыш",
            "концовка", "рэп", "хук", "переход", "часть")
ALL = {"all", "everyone", "everybody", "together", "both", "group", "tutti", "unison", "все", "вместе", "оба", "обе",
       "хор", "хором", "всё"}
JUNK = re.compile(r"^(you might also like|see .* live|get tickets)|^\d+\s+contributors|lyrics$", re.I)
SPLIT = re.compile(r"\s*(?:,|&|\+|/|;|\bx\b|×|\band\b|\bfeat\.?|\bft\.?|\bи\b|\bwith\b)\s*", re.I)


def norm(w):
    w = unicodedata.normalize("NFKC", w).lower().replace("ё", "е")
    return "".join(ch for ch in w if ch.isalnum())


# ---------- formatted text -> lines of (char, style) ----------
# Genius tells who sings a line by its formatting: names in "[Verse 1: Karina, <i>Winter</i>, <b>Both</b>]" are
# written in the style of their lines. The text comes from the tab's editor as HTML (pasted with formatting), or
# as plain text with Markdown marks (**bold**, *italic*, __underline__).
BOLD, ITAL, UND = "b", "i", "u"
BLOCK = {"div", "p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section", "article"}
TAG_STYLE = {"b": BOLD, "strong": BOLD, "i": ITAL, "em": ITAL, "u": UND, "ins": UND}


def _css(style):
    """Inline CSS -> {flag: on/off} (pages copied from a browser carry formatting as styles)."""
    out = {}
    for decl in (style or "").split(";"):
        if ":" not in decl:
            continue
        k, v = (t.strip().lower() for t in decl.split(":", 1))
        if k == "font-weight":
            out[BOLD] = v in ("bold", "bolder") or (v.isdigit() and int(v) >= 600)
        elif k == "font-style":
            out[ITAL] = v in ("italic", "oblique")
        elif k in ("text-decoration", "text-decoration-line"):
            out[UND] = "underline" in v
    return out


def _html_lines(text):
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.lines, self.cur, self.stack, self.skip = [], [], [], 0

        def style(self):
            st = {}
            for _, d in self.stack:
                st.update(d)
            return frozenset(k for k, v in st.items() if v)

        def newline(self):
            self.lines.append(self.cur)
            self.cur = []

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "head", "title"):
                self.skip += 1
            if tag == "br":
                self.newline()
                return
            if tag in BLOCK and self.cur:
                self.newline()
            d = {TAG_STYLE[tag]: True} if tag in TAG_STYLE else {}
            d.update(_css(dict(attrs).get("style")))
            if tag not in ("img", "hr", "input", "meta", "link", "wbr"):
                self.stack.append((tag, d))

        def handle_endtag(self, tag):
            if tag in ("script", "style", "head", "title"):
                self.skip = max(0, self.skip - 1)
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break
            if tag in BLOCK and self.cur:
                self.newline()

        def handle_data(self, data):
            if self.skip:
                return
            st = self.style()
            self.cur += [(ch if ch not in "\r\n\t" else " ", st) for ch in data]

    p = P()
    p.feed(text)
    p.close()
    if p.cur:
        p.newline()
    return p.lines


# a mark touches a word on one side only ("f**k" is not bold)
MD = re.compile(r"(?<![^\W_])(?:\*{1,3}|_{1,2})|(?:\*{1,3}|_{1,2})(?![^\W_])")


def _markdown(lines):
    """Turn **bold** / *italic* / __underline__ marks into styles (on top of HTML formatting)."""
    on = set()
    out = []
    for line in lines:
        text = "".join(c for c, _ in line)
        if re.fullmatch(r"\s*\[.*\]\s*", re.sub(MD, "", text)) and not any(st for _, st in line):
            on = set()  # a section header starts with no marks open
        res, pos = [], 0
        for m in MD.finditer(text):
            res += [(c, st | frozenset(on)) for c, st in line[pos:m.start()]]
            mk = m.group()
            flags = {"***": (BOLD, ITAL), "**": (BOLD,), "*": (ITAL,), "__": (UND,), "_": (ITAL,)}[mk]
            for f in flags:
                on.symmetric_difference_update({f})
            pos = m.end()
        res += [(c, st | frozenset(on)) for c, st in line[pos:]]
        out.append(res)
    return out


def styled_lines(text):
    text = text or ""
    if re.search(r"<\s*(br|b|i|u|em|strong|div|p|span)\b", text, re.I):
        lines = _html_lines(text)
    else:
        lines = [[(c, frozenset()) for c in ln] for ln in text.splitlines()]
    return _markdown(lines)


def _style_of(line, a, b):
    sts = [st for c, st in line[a:b] if not c.isspace()]
    return max(set(sts), key=sts.count) if sts else frozenset()


def _names(s, offset=0):
    """Names in a header part with their positions: 'Karina, Winter & Giselle' -> [(name, start, end)]."""
    out, pos = [], 0
    for m in list(SPLIT.finditer(s)) + [None]:
        end = m.start() if m else len(s)
        raw = s[pos:end]
        name = re.sub(r"\(.*?\)", "", re.sub(r"[*_`\"]", "", raw)).strip()
        if name:
            lead = len(raw) - len(raw.lstrip())
            out.append((name, offset + pos + lead, offset + pos + lead + len(raw.strip())))
        pos = m.end() if m else len(s)
    return out


def _header(h, offset):
    """'Verse 1: Karina, Winter' -> (section, [(name, start, end)] or None)."""
    if ":" in h:
        sec, who = h.split(":", 1)
        return sec.strip(), _names(who, offset + len(sec) + 1) or None
    low = h.lower()
    if any(low.startswith(k) or k in low.split() for k in SECTIONS):
        return h.strip(), None
    return "", _names(h, offset) or None  # "[Karina]"


def _base(section):
    return re.sub(r"[\d\s.#№]+$", "", section.strip().lower())


def _group_names(lines, song_artist):
    """The song's artist written in a tag means "everybody" only for a group: "[Chorus: aespa]" next to tags with
    the members' names. For a solo artist with guests ("Damiano David feat. Tyla": "[Verse 1: Damiano David]",
    "[Chorus: Damiano David & Tyla]") the same name is a singer. So it is the group when it never shares a tag
    with other names and at least two other singers are named."""
    key = norm(song_artist or "")
    if not key:
        return set()
    others, shared = set(), False
    for line in lines:
        m = re.fullmatch(r"\s*\[(.+?)\]\s*", "".join(c for c, _ in line))
        if not m:
            continue
        _, names = _header(m.group(1), m.start(1))
        keys = {norm(n) for n, _, _ in names or []} - {""} - ALL
        if key in keys and len(keys) > 1:
            shared = True
        others |= keys - {key}
    return {key} if not shared and len(others) >= 2 else set()


def parse(text, song_artist=None):
    """-> (words, artists): words = [(normalized word, singers)] with singers = tuple of names, ("*all*",) or None;
    artists = individual singers in order of appearance (normalized key -> display name).
    text: plain, Markdown-marked, or HTML (formatting tells which of the section's singers sings a line)."""
    words, artists, by_section = [], {}, {}
    lines = styled_lines(text)
    group = _group_names(lines, song_artist)
    cur, by_style = None, {}
    for line in lines:
        txt = "".join(c for c, _ in line)
        m = re.fullmatch(r"\s*\[(.+?)\]\s*", txt)
        if m:
            sec, names = _header(m.group(1), m.start(1))
            if names:
                keys, by_style = [], {}
                for n, a, b in names:
                    k = norm(n)
                    if not k:
                        continue
                    k = "*all*" if k in ALL or k in group else k
                    if k != "*all*":
                        artists.setdefault(k, n)
                    st = _style_of(line, a, b)
                    by_style[st] = tuple(dict.fromkeys(by_style.get(st, ()) + (k,)))
                    if k not in keys:
                        keys.append(k)
                indiv = tuple(k for k in keys if k != "*all*")
                cur = indiv or (("*all*",) if keys else None)
                if sec:
                    by_section[_base(sec)] = (cur, by_style)
            else:
                cur, by_style = by_section.get(_base(sec), (None, {}))
            continue
        clean = re.sub(r"\d*Embed\s*$", "", txt)  # Genius glues "Embed" to the last line
        if not clean.strip() or JUNK.search(clean.strip()):
            continue
        # (ad-libs in brackets) are usually someone else / the backing: labelled only if their formatting says who
        depth = 0
        for t in re.finditer(r"\(|\)|[^\s()]+", clean):
            tok = t.group()
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth = max(0, depth - 1)
            else:
                w = norm(tok)
                if not w:
                    continue
                st = _style_of(line, t.start(), t.end())
                if depth:
                    who = by_style.get(st) if st else None
                else:
                    who = by_style.get(st, cur)
                if who and "*all*" in who:
                    who = ("*all*",)
                words.append((w, who))
    return words, artists


def align(lyric_words, asr_words):
    """Singers for each Whisper word (same order as asr_words): exact matches first, then similar words in the
    gaps; words in between take the singers of the lyric part they fall in (when it is unambiguous)."""
    a = [norm(w["word"]) for w in asr_words]
    b = [w for w, _ in lyric_words]
    match = [None] * len(a)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = sm.get_matching_blocks()
    prev_a = prev_b = 0
    for blk in blocks:
        # fuzzy pairs inside the gap before this block, in order
        j = prev_b
        for i in range(prev_a, blk.a):
            for jj in range(j, min(blk.b, j + 4)):
                if a[i] and difflib.SequenceMatcher(None, a[i], b[jj]).ratio() >= 0.75:
                    match[i], j = jj, jj + 1
                    break
        for t in range(blk.size):
            match[blk.a + t] = blk.b + t
        prev_a, prev_b = blk.a + blk.size, blk.b + blk.size
    out, n_matched = [], sum(m is not None for m in match)
    anchors = [(i, m) for i, m in enumerate(match) if m is not None]
    for i in range(len(a)):
        if match[i] is not None:
            out.append(lyric_words[match[i]][1])
            continue
        lo = max([m for k, m in anchors if k < i], default=0)
        hi = min([m for k, m in anchors if k > i], default=len(b) - 1)
        span = {lyric_words[j][1] for j in range(lo, hi + 1)} if b else set()
        out.append(span.pop() if len(span) == 1 else None)
    return out, (n_matched / len(a) if a else 0.0)
