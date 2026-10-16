"""Prompt strata for the PF analysis: does the prompt constrain the subject?

Deterministic rule on an en_core_web_sm parse (`classify`, the primary
classification of the appendix). The subject is found with the spaCy stripper's mention finder
(`SpacySubjectStripper.find_mentions`); if no mention is found, the subject is
the root of the first noun chunk whose root is not an image-medium noun
("photo", "painting", ...), so every prompt is classified.

  action/pose  the subject governs a verbal clause (a dependent VERB with dep
               acl or relcl: "a kitten chasing a butterfly", "a dog sitting"),
               or is the nominal subject of a verb ("a kitten sleeps")
  attribute    the subject carries a modifier that is not part of its name:
               an adjectival or compound pre-modifier (amod, compound, nmod),
               or a "with"-phrase attached to it ("a kitten with big eyes")
  subject-dependent = action/pose OR attribute (the two overlap; a prompt in
               both counts in both sub-strata and once in subject-dependent)
  scene-only   neither
"""

from __future__ import annotations

from masc.text import SpacySubjectStripper

MEDIUM_NOUNS = {
    "photo",
    "photograph",
    "picture",
    "image",
    "painting",
    "illustration",
    "sketch",
    "drawing",
    "render",
    "rendering",
    "art",
    "artwork",
    "portrait",
    "watercolor",
    "style",
    "scene",
    "shot",
    "close",
    "view",
    "poster",
    "design",
    "version",
}


def _subject_head(stripper: SpacySubjectStripper, doc, object_name: str):
    spans = stripper.find_mentions(doc, object_name)
    if spans:
        return [doc[s:e].root for s, e in spans], "mention"
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() not in MEDIUM_NOUNS:
            return [chunk.root], "first_noun_chunk"
    return [], "none"


def classify(stripper: SpacySubjectStripper, prompt: str, object_name: str) -> dict:
    doc = stripper.nlp(prompt)
    heads, how = _subject_head(stripper, doc, object_name)
    name_lower = {t.lower_ for t in stripper.nlp(object_name)}
    action, attribute, evidence = False, False, []
    for h in heads:
        for c in h.children:
            if c.dep_ in ("acl", "relcl") and c.pos_ in ("VERB", "AUX"):
                action = True
                evidence.append(f"verb:{c.text}")
            if c.dep_ in ("amod", "compound", "nmod") and c.i < h.i and c.lower_ not in name_lower:
                attribute = True
                evidence.append(f"mod:{c.text}")
            if c.dep_ == "prep" and c.lower_ == "with":
                attribute = True
                evidence.append("with-phrase")
        if h.dep_ in ("nsubj", "nsubjpass") and h.head.pos_ == "VERB":
            action = True
            evidence.append(f"subj-of:{h.head.text}")
    return {
        "action_pose": action,
        "attribute": attribute,
        "subject_dependent": action or attribute,
        "scene_only": not (action or attribute),
        "subject_found_by": how,
        "subject_tokens": " ".join(h.text for h in heads),
        "evidence": ";".join(evidence),
    }


def classify_lexical(stripper: SpacySubjectStripper, prompt: str, object_name: str) -> dict:
    """Lexical classification, the appendix's sensitivity check: the
    dependency rule misses participles that en_core_web_sm attaches to the
    wrong head (e.g. "a steam locomotive releasing steam clouds" parses with
    "clouds" as the head). action/pose = the prompt contains a non-auxiliary
    VERB other than medium verbs ("styled", "rendered"); attribute as in
    `classify`."""
    base = classify(stripper, prompt, object_name)
    doc = stripper.nlp(prompt)
    verbs = [
        t.text
        for t in doc
        if t.pos_ == "VERB"
        and t.lemma_.lower()
        not in {
            "style",
            "render",
            "depict",
            "show",
            "feature",
            "capture",
            "inspire",
            "focus",
            "emphasize",
        }
    ]
    action = bool(verbs)
    return {
        **{f"{k}": v for k, v in base.items()},
        "action_pose": action,
        "subject_dependent": action or base["attribute"],
        "scene_only": not (action or base["attribute"]),
        "evidence": ";".join(verbs),
    }
