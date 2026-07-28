"""
WEMA — Women's Emergency Medical AI
src/prompt.py
"""

import re

# Each fallback (except not_breathing, which is always about the baby) has a
# "self" variant (caller is the patient: "you"/"your") and an "other" variant
# (caller is reporting on someone else, e.g. "my wife": "her"/"help her") --
# mirrors the caller-vs-patient distinction rag.py's SYSTEM prompt already
# makes for real LLM generations. Found missing here via the 68-scenario
# evaluation (S057: "my wife... she don faint" was answered as if the caller
# were the patient) -- the keyword-routed fallback path had no such check.
_EMERGENCY_FALLBACKS = {
    "bleeding": {
        "self": (
            "Stay calm, I am here with you. "
            "Massage your lower belly firmly in circles until it feels hard. "
            "Then put your baby to your breast — this helps slow the bleeding. "
            "Lie flat and keep warm. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. "
            "Massage her lower belly firmly in circles until it feels hard. "
            "Then put the baby to her breast — this helps slow the bleeding. "
            "Help her lie flat and keep her warm. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "bleeding_pregnant": {
        "self": (
            "Stay calm, I am here with you. "
            "Lie down right now and do not press on your belly. "
            "Keep warm and stay as still as you can. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. "
            "Help her lie down right now and do not press on her belly. "
            "Keep her warm and as still as possible. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "fits": {
        "self": (
            "Stay calm. Lie on your left side right now. "
            "Do not put anything in your mouth. "
            "Someone should stay with you. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm. Lay her on her left side right now. "
            "Do not put anything in her mouth and do not hold her down. "
            "Stay with her. "
            "Help is being alerted. Get to a hospital now."
        ),
    },
    "cord": {
        "self": (
            "Get on your hands and knees with your chest down and hips up right now. "
            "Do not push the cord back in and do not stand up. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Help her get on her hands and knees with her chest down and hips up right now. "
            "Do not push the cord back in and do not let her stand up. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "not_breathing": (
        "Dry the baby quickly with a clean cloth and rub the back firmly. "
        "Keep the baby warm. "
        "If still not breathing, give gentle puffs covering the mouth and nose. "
        "Help is being alerted. Get to a hospital now."
    ),
    # Always about the baby, regardless of who is calling -- unlike "fits"
    # (adult seizure), where the caller-vs-patient framing matters. Matches
    # rag.py's SYSTEM prompt protocol for newborn convulsions. Previously a
    # baby convulsing was matched by the same "shake"/"convuls" keywords as
    # an adult seizure and got the adult "fits" template (addressing the
    # caller as if she were the one seizing) -- found via S064 in the
    # 68-scenario evaluation.
    "newborn_convulsions": (
        "Turn the baby gently onto their side right now. "
        "Do not shake or restrain the baby. Keep the baby warm and flat. "
        "Help is being alerted. Get to a hospital now."
    ),
    "ectopic": {
        "self": (
            "Lie flat right now. Do not get up and do not press your belly. "
            "Help is being alerted. Get to a hospital now by the fastest way possible."
        ),
        "other": (
            "Help her lie flat right now. Do not let her get up and do not press her belly. "
            "Help is being alerted. Get her to a hospital now by the fastest way possible."
        ),
    },
    # Below: expanded from the original 6 categories to cover all 17 emergency
    # types from the 68-scenario evaluation, reusing the exact clinician-reviewed
    # wording already in rag.py's SYSTEM prompt / data/pdfs/WEMA_Clinical_Action_Protocol.md
    # -- no new clinical content, just routing to text that was already approved.
    # Water/sipping is only included where the category itself implies the
    # caller is conscious and able to describe her own symptoms (sepsis,
    # anaemia, sickle cell, conscious malaria) -- never in "default" or any
    # confused/unconscious-leaning category, matching the SYSTEM prompt's own
    # care around aspiration risk.
    "wound_bleeding": {
        "self": (
            "Stay calm, I am here with you. "
            "Press firmly on the wound with the cleanest cloth you have. "
            "Do not remove it once it is on — add more cloth on top if it soaks through. "
            "Do not massage your belly. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. "
            "Press firmly on the wound with the cleanest cloth available. "
            "Do not remove it once it is on — add more cloth on top if it soaks through. "
            "Do not massage her belly. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "retained_placenta": {
        "self": (
            "Stay calm, I am here with you. "
            "Do not massage your belly and do not pull the cord. "
            "Keep lying still and empty your bladder by urinating. "
            "Do not try to remove the placenta yourself. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. "
            "Do not massage her belly and do not pull the cord. "
            "Keep her lying still and help her empty her bladder. "
            "Do not try to remove the placenta. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "mastitis": {
        "self": (
            "I am here with you. This sounds like a breast infection. "
            "Keep breastfeeding or expressing milk from that breast — it helps clear it. "
            "Put a warm cloth on it before feeding. "
            "Help is being alerted. Get to a facility today for treatment."
        ),
        "other": (
            "I am here with you. This sounds like a breast infection. "
            "She should keep breastfeeding or expressing milk from that breast — it helps clear it. "
            "A warm cloth before feeding will help. "
            "Help is being alerted. Get her to a facility today for treatment."
        ),
    },
    "shoulder_dystocia": {
        "self": (
            "Stay calm, this can be managed. Pull your knees up firmly toward your chest right now. "
            "Have someone press firmly just above your pubic bone. "
            "Do not pull on the baby. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, this can be managed. Help her pull her knees up firmly toward her chest right now. "
            "Press firmly just above her pubic bone. "
            "Do not pull on the baby. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "obstructed_labour": {
        "self": (
            "Stay calm, I am here with you. Lie on your left side right now and stop pushing. "
            "Breathe slowly through each contraction. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. Help her lie on her left side right now and stop pushing. "
            "She should breathe slowly through each contraction. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "preterm_labour": {
        "self": (
            "Stay calm, I am here with you. Lie on your left side right now. Do not push. "
            "Breathe slowly through each contraction. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "Stay calm, I am here with you. Help her lie on her left side right now. She should not push. "
            "Breathing slowly through each contraction will help. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "sepsis_pregnant": {
        "self": (
            "I am here with you. Lie on your left side right now and keep warm. "
            "Sip some water if you can. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie on her left side right now and keep her warm. "
            "Small sips of water will help. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "sepsis_postpartum": {
        "self": (
            "I am here with you. Lie on your left side right now and keep warm. "
            "Do not touch any wound you have. Sip some water if you can. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie on her left side right now and keep her warm. "
            "Do not touch any wound she has. Small sips of water will help. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "severe_anaemia": {
        "self": (
            "I am here with you. Lie on your left side right now and rest completely. "
            "Sip some water if you can and do not exert yourself. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie on her left side right now — she needs to rest completely. "
            "Small sips of water will help, and she should not exert herself. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "sickle_cell": {
        "self": (
            "I am here with you. Lie down and rest right now. Keep warm. "
            "Drink water if you are able to. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie down and rest right now. Keep her warm. "
            "She should drink water if she is able to. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "malaria_conscious": {
        "self": (
            "I am here with you. Lie on your left side right now and keep warm. "
            "Sip some water. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie on her left side right now and keep her warm. "
            "Small sips of water will help. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "malaria_confused": {
        "self": (
            "I am here with you. Lie on your left side right now and keep warm. "
            "Do not eat or drink anything right now. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her lie on her left side right now and keep her warm. "
            "Do not give her anything to eat or drink right now. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "gestational_diabetes_high": {
        "self": (
            "I am here with you. Sit or lie on your side, whichever feels safest, right now. "
            "Do not eat or drink anything if you feel too unwell to swallow safely. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. Help her sit or lie on her side, whichever is safest, right now. "
            "Do not give her anything to eat or drink if she is not responding well. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "gestational_diabetes_low": {
        "self": (
            "I am here with you. Sit down right now and eat or drink something sweet immediately. "
            "After that, arrange transport. "
            "Help is being alerted."
        ),
        "other": (
            "I am here with you. Help her sit down right now and give her something sweet to eat or drink immediately. "
            "After that, arrange transport. "
            "Help is being alerted."
        ),
    },
    "hyperemesis": {
        "self": (
            "I am here with you. Take only tiny sips of water right now — no solid food, no other drinks. "
            "Lie on your side. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. She should take only tiny sips of water right now — no solid food, no other drinks. "
            "Help her lie on her side. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "miscarriage_heavy": {
        "self": (
            "I know this is frightening, I am here with you. Keep any tissue you pass in a clean bag to show the doctor. "
            "Lie down and do not press on your belly. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I know this is frightening, I am here with you. Keep any tissue she passes in a clean bag to show the doctor. "
            "Help her lie down and do not press on her belly. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
    "miscarriage_light": {
        "self": (
            "I am here with you. Rest at home and lie down right now. "
            "Monitor your bleeding, and do not massage your belly. "
            "If bleeding gets much heavier or pain becomes severe, go to a health facility immediately."
        ),
        "other": (
            "I am here with you. She should rest at home and lie down right now. "
            "Keep an eye on her bleeding, and do not massage her belly. "
            "If bleeding gets much heavier or pain becomes severe, get her to a health facility immediately."
        ),
    },
    "default": {
        "self": (
            "I am here with you. "
            "Lie on your left side, keep warm, and do not move. "
            "Help is being alerted. Get to a hospital now."
        ),
        "other": (
            "I am here with you. "
            "Help her lie on her left side, keep her warm, and do not let her move. "
            "Help is being alerted. Get her to a hospital now."
        ),
    },
}

_CONVERSATIONAL_RESPONSES = {
    "greeting": [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "good night", "how are you"
    ],
    "thanks": [
        "thank you", "thanks", "thank u", "thank", "God bless"
    ],
    "ok": [
        "ok", "okay", "alright", "sure", "yes", "yeah", "yep"
    ],
}


def is_conversational(text: str) -> str | None:
    text_lower = text.lower().strip()
    for intent, phrases in _CONVERSATIONAL_RESPONSES.items():
        if any(text_lower.startswith(p) or text_lower == p for p in phrases):
            return intent
    return None


def get_conversational_response(intent: str) -> str:
    responses = {
        "greeting": (
            "Hello, I am here with you. "
            "Please tell me what is happening right now so I can help you."
        ),
        "thanks": (
            "You are welcome. Please stay safe. "
            "Call WEMA anytime you need help."
        ),
        "ok": (
            "I am here. Please tell me what is happening so I can help you."
        ),
    }
    return responses.get(intent, responses["ok"])


# Pidgin "no fit" / "no dey fit" means "cannot" — it must NOT route to the
# seizure response. Strip it before checking for "fit" as a seizure word,
# and require a word boundary so "profit"/"outfit"/"benefit" never match.
_PIDGIN_CANNOT = re.compile(r"\bno\s+(dey\s+)?fit\b")
_FIT_SEIZURE = re.compile(r"\bfit(s|ting)?\b")

# Belly massage is correct for bleeding AFTER BIRTH only. For bleeding in
# pregnancy (placenta praevia etc.) pressing the belly is dangerous, so the
# massage response is gated on an explicit birth mention. Includes Pidgin
# "I born" (= I gave birth).
_BIRTH_MENTION = re.compile(r"\bbirth\b|\bborn\b|\bdeliver(ed|y)?\b|\bcaesarean\b|\bc-section\b|\bcesarean\b", re.IGNORECASE)

# "blood" must not match "blood pressure" -- a caller mentioning known high
# blood pressure while describing a seizure (very common eclampsia phrasing,
# e.g. "Blood pressure high dem tell me last week") was being misrouted to
# the bleeding-in-pregnancy fallback instead of the seizure fallback, silently
# dropping the seizure-specific safety instructions (left-side position,
# nothing in the mouth). Found via tests/S014 in the 68-scenario evaluation.
_BLOOD_NOT_PRESSURE = re.compile(r"\bblood\b(?!\s*(pressure|level))", re.IGNORECASE)


def _mentions_seizure_fit(text: str) -> bool:
    cleaned = _PIDGIN_CANNOT.sub(" ", text)
    return bool(_FIT_SEIZURE.search(cleaned))


# Distinguishes "the baby is convulsing" from "the (adult) patient is having a
# seizure" -- both match the same "shake"/"convuls"/"fit" keywords otherwise.
_BABY_MENTION = re.compile(r"\bbaby\b|\bnewborn\b|\binfant\b|\bpikin\b", re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def _baby_is_the_one_convulsing(text: str) -> bool:
    """A baby-mention and a shake/convuls keyword ANYWHERE in the text is not
    enough -- e.g. 'I don dey shake and vomit ... My baby no dey kick' is a
    mother's own malaria symptoms plus an unrelated reduced-fetal-movement
    mention, not a convulsing newborn, but both keywords are present. Require
    them in the same sentence, so the baby-mention is actually the subject of
    the shaking/convulsing. Found via S052 in the 68-scenario evaluation,
    misrouted to the newborn-convulsions response after the S064 fix."""
    for sentence in _SENTENCE_SPLIT.split(text):
        if _BABY_MENTION.search(sentence) and (
            _mentions_seizure_fit(sentence) or any(w in sentence for w in ["convuls", "shake", "shaking"])
        ):
            return True
    return False


# Mirrors rag.py's SYSTEM prompt distinction: "my wife/sister/daughter/mother
# is..." or "she is/just/has..." means the caller is NOT the patient. Defaults
# to "self" when ambiguous, matching the previous single-variant behaviour.
_THIRD_PARTY_CALLER = re.compile(
    r"\bmy\s+(wife|sister|daughter|mother|mum|mummy|friend|neighbour|neighbor)\b"
    r"|\bshe\s+(is|just|has|don|dey)\b",
    re.IGNORECASE,
)


def _fallback_variant(entry, caller_text: str) -> str:
    if isinstance(entry, str):
        return entry
    return entry["other"] if _THIRD_PARTY_CALLER.search(caller_text) else entry["self"]


# "Lost blood" (past tense, e.g. "I lost a lot of blood during delivery") describes
# blood already lost, not bleeding happening now -- distinct from "bleeding"/"still
# bleeding". The "bleeding" template's belly-massage instruction is for active
# bleeding and is not safe general advice for someone now showing anaemia/shock
# symptoms (racing heart, breathlessness, weakness) from a past blood loss. Found
# via S049 in the 68-scenario evaluation (surfaced when the main LLM path failed
# and fell through to this keyword fallback).
_PAST_BLOOD_LOSS_ONLY = re.compile(r"\blost\s+(a\s+lot\s+of\s+)?blood\b", re.IGNORECASE)

_WOUND_MENTION = re.compile(r"\bcut\b|\btear\b|\bwound\b", re.IGNORECASE)
_ACTIVELY_BLEEDING = re.compile(r"\bbleed|\bblood|\bstop\b", re.IGNORECASE)
_PLACENTA_NOT_OUT = re.compile(r"\bplacenta\b", re.IGNORECASE)
_NOT_DELIVERED = re.compile(r"not\s+come|not\s+out|still\s+inside|hasn.t\s+come", re.IGNORECASE)
_BREAST_MENTION = re.compile(r"\bbreast", re.IGNORECASE)
_BREAST_INFECTION_SIGNS = re.compile(r"\b(red|hard|swollen|painful)\b", re.IGNORECASE)
_SHOULDER_STUCK = re.compile(r"\bshoulder", re.IGNORECASE)
_LABOUR_STUCK_SIGNS = re.compile(
    r"not\s+come\s+out|not\s+coming|no\s+wan\s+come|will\s+not\s+come|stuck|not\s+moving\s+down",
    re.IGNORECASE,
)
_LABOUR_CONTEXT = re.compile(r"\blabour\b|\blabor\b|\bpush(ing)?\b|\bcontraction", re.IGNORECASE)
_SICKLE_CELL_MENTION = re.compile(r"sickle\s*cell", re.IGNORECASE)
_MALARIA_MENTION = re.compile(r"\bmalaria\b", re.IGNORECASE)
_VOMIT_MENTION = re.compile(r"\bvomit", re.IGNORECASE)
_STRONG_VOMIT_SIGNS = re.compile(r"cannot\s+stop\s+vomiting|can.?t\s+stop\s+vomiting|keep.{0,10}down|kept.{0,10}down", re.IGNORECASE)
# "not respond" alone missed Pidgin phrasing like "she no dey respond to me"
# (no/not + optional dey + respond) -- found via S057, where a caller
# describing an unresponsive, fainted woman was routed to a fallback that
# told her to eat/drink something sweet, an aspiration risk for someone who
# may not be able to swallow safely. Real safety bug, not just a wording gap.
_CONFUSED_OR_UNCONSCIOUS = re.compile(
    r"\bconfus|\bunconscious|not\s+respond|no\s+(dey\s+)?respond|can.?t\s+open.*eye|cannot\s+open.*eye|"
    r"not\s+wak(e|ing)|\bfaint(ed)?\b",
    re.IGNORECASE,
)
_FEVER_SIGNS = re.compile(r"\bfever\b|hot\s+like\s+fire|hot\s+body|high\s+temperature", re.IGNORECASE)
_ANAEMIA_SIGNS = re.compile(
    r"cannot\s+breathe|can.?t\s+breathe|breathless|heart.{0,15}(fast|racing|race)|tired\s+all\s+the\s+time|"
    r"tire\s+well\s+well",
    re.IGNORECASE,
)
_SHAKE_WITH_FEVER = re.compile(r"shak(e|ing)?.{0,25}fever|fever.{0,25}shak(e|ing)?", re.IGNORECASE)
_SUGAR_MENTION = re.compile(r"\bsugar\b|\bdiabetes\b", re.IGNORECASE)
_PRETERM_SIGNS = re.compile(
    r"\bwater[s]?\b.{0,15}\b(broke|burst|break)\b|contraction|labour\s+pain|labor\s+pain|pains?\s+every",
    re.IGNORECASE,
)
_MISCARRIAGE_TISSUE = re.compile(r"\btissue\b|\bclot", re.IGNORECASE)
_MISCARRIAGE_PASSING = re.compile(r"\bpass(ing)?\b", re.IGNORECASE)
_MISCARRIAGE_LIGHT_SIGNS = re.compile(r"\bsmall\b.{0,20}\b(blood|cramp)|doctor\s+say.{0,20}rest", re.IGNORECASE)


def get_emergency_fallback(caller_text: str) -> str:
    text = caller_text.lower()

    if (_PAST_BLOOD_LOSS_ONLY.search(text)
            and "bleed" not in text and "haemorrhage" not in text and "hemorrhage" not in text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["severe_anaemia"], caller_text)

    # Checked before the generic bleed/labour/fever checks below because each
    # of these can co-occur with "bleed"/labour/fever words while needing a
    # different, more specific response (e.g. a retained placenta must NOT
    # be massaged, unlike ordinary postpartum bleeding; an infected-but-not-
    # actively-bleeding wound needs sepsis care, not direct pressure).
    # Miscarriage is by definition before birth -- a birth-mention here means
    # this is postpartum bleeding/clots (e.g. retained products), not a
    # miscarriage, even though both can present as "passing clots".
    if not _BIRTH_MENTION.search(text):
        if _MISCARRIAGE_TISSUE.search(text) and _MISCARRIAGE_PASSING.search(text):
            return _fallback_variant(_EMERGENCY_FALLBACKS["miscarriage_heavy"], caller_text)
        if _MISCARRIAGE_LIGHT_SIGNS.search(text):
            return _fallback_variant(_EMERGENCY_FALLBACKS["miscarriage_light"], caller_text)
    if _BIRTH_MENTION.search(text) and _WOUND_MENTION.search(text) and _ACTIVELY_BLEEDING.search(text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["wound_bleeding"], caller_text)
    if _PLACENTA_NOT_OUT.search(text) and _NOT_DELIVERED.search(text) and "praevia" not in text:
        return _fallback_variant(_EMERGENCY_FALLBACKS["retained_placenta"], caller_text)
    if _BREAST_MENTION.search(text) and _BREAST_INFECTION_SIGNS.search(text) and _WOUND_MENTION.search(text) is None:
        return _fallback_variant(_EMERGENCY_FALLBACKS["mastitis"], caller_text)
    if _SHOULDER_STUCK.search(text) and "stuck" in text:
        return _fallback_variant(_EMERGENCY_FALLBACKS["shoulder_dystocia"], caller_text)
    if _LABOUR_CONTEXT.search(text) and _LABOUR_STUCK_SIGNS.search(text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["obstructed_labour"], caller_text)

    # Fever takes priority over generic bleed/anaemia/preterm checks below --
    # sepsis is more urgent and specific than any of those when fever is
    # explicitly present. Checked before "bleed" so an infected wound/fever
    # case isn't misread as active bleeding, and before "tired"/"water broke"
    # so anaemia/preterm-labour signs don't swallow a febrile presentation.
    if _SICKLE_CELL_MENTION.search(text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["sickle_cell"], caller_text)
    if _MALARIA_MENTION.search(text) or _SHAKE_WITH_FEVER.search(text) or (
        _VOMIT_MENTION.search(text) and "shak" in text and not _mentions_seizure_fit(text)
    ):
        if _CONFUSED_OR_UNCONSCIOUS.search(text):
            return _fallback_variant(_EMERGENCY_FALLBACKS["malaria_confused"], caller_text)
        return _fallback_variant(_EMERGENCY_FALLBACKS["malaria_conscious"], caller_text)
    if _FEVER_SIGNS.search(text):
        if _BIRTH_MENTION.search(text) or "abortion" in text:
            return _fallback_variant(_EMERGENCY_FALLBACKS["sepsis_postpartum"], caller_text)
        return _fallback_variant(_EMERGENCY_FALLBACKS["sepsis_pregnant"], caller_text)

    if "bleed" in text or "haemorrhage" in text or "hemorrhage" in text or _BLOOD_NOT_PRESSURE.search(text):
        if _BIRTH_MENTION.search(text):
            return _fallback_variant(_EMERGENCY_FALLBACKS["bleeding"], caller_text)
        return _fallback_variant(_EMERGENCY_FALLBACKS["bleeding_pregnant"], caller_text)

    # Anaemia/cardiac signs checked before ectopic below -- "I might collapse"
    # from breathlessness/racing heart is a different emergency than fainting
    # from one-sided abdominal pain, but both could match a bare "collapse".
    if _ANAEMIA_SIGNS.search(text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["severe_anaemia"], caller_text)

    # Sugar/diabetes checked before ectopic's "faint" trigger below -- a
    # diabetic caller who mentions fainting needs the diabetes response, not
    # the ectopic one, even though both can present with fainting.
    if _SUGAR_MENTION.search(text):
        if _CONFUSED_OR_UNCONSCIOUS.search(text) or "dizzy" in text or "not eaten" in text or "did not eat" in text:
            return _fallback_variant(_EMERGENCY_FALLBACKS["gestational_diabetes_high"], caller_text)
        return _fallback_variant(_EMERGENCY_FALLBACKS["gestational_diabetes_low"], caller_text)

    _shaking_or_seizure = _mentions_seizure_fit(text) or any(w in text for w in ["convuls", "shake", "seizure", "shaking"])
    if _shaking_or_seizure and _baby_is_the_one_convulsing(text):
        return _EMERGENCY_FALLBACKS["newborn_convulsions"]
    if _shaking_or_seizure:
        return _fallback_variant(_EMERGENCY_FALLBACKS["fits"], caller_text)
    if any(w in text for w in ["cord", "rope", "string", "umbilical"]):
        return _fallback_variant(_EMERGENCY_FALLBACKS["cord"], caller_text)
    if any(w in text for w in ["not breathing", "no breath", "baby not", "not cry", "not crying",
                               "no dey breathe", "no cry", "pikin no"]):
        return _EMERGENCY_FALLBACKS["not_breathing"]
    if any(w in text for w in ["one side", "sharp pain", "sharp cramp", "shoulder tip",
                                "ectopic", "collapse", "collapsed", "faint", "fainted"]):
        return _fallback_variant(_EMERGENCY_FALLBACKS["ectopic"], caller_text)

    if _STRONG_VOMIT_SIGNS.search(text) or (_VOMIT_MENTION.search(text) and "swoll" not in text and "headache" not in text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["hyperemesis"], caller_text)

    if not _BIRTH_MENTION.search(text) and _PRETERM_SIGNS.search(text):
        return _fallback_variant(_EMERGENCY_FALLBACKS["preterm_labour"], caller_text)

    return _fallback_variant(_EMERGENCY_FALLBACKS["default"], caller_text)


def get_fallback_response(reason: str = "api_down") -> str:
    fallbacks = {
        "api_down": (
            "I am here with you. "
            "Please go to your nearest hospital right now. "
            "If you are bleeding, press firmly on your lower belly and lie flat. "
            "If someone had a seizure, place her on her left side. "
            "Help is being alerted to you now."
        ),
        "no_results": (
            "I want to help you. "
            "Please go to your nearest hospital immediately — do not wait. "
            "If you are bleeding, press on your lower belly and keep warm. "
            "Help is being alerted to you now."
        ),
        "timeout": (
            "I am still here. "
            "The most important thing right now is to get to a hospital. "
            "Go immediately — do not wait. "
            "Help is being alerted to you now."
        ),
    }
    return fallbacks.get(reason, fallbacks["api_down"])


def get_stt_retry_prompt() -> str:
    return "I did not hear you clearly. Please speak again and tell me what is happening."


def get_greeting() -> str:
    return (
        "Hello, this is WEMA. "
        "I am not a doctor, and this call is recorded. "
        "I am here to help you — please tell me what is happening."
    )
