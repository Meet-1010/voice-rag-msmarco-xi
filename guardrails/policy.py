"""Refusal taxonomy.

Every refusal carries a machine-readable reason code and a message in the user's
language. A system that refuses without saying why is indistinguishable from one
that is broken, and the UI surfaces the code so a demo can show the difference.
"""
from __future__ import annotations

from harness.schemas import ReasonCode

_MESSAGES = {
    ReasonCode.EMPTY_INPUT: {
        "en": "I did not catch a question there. Could you say that again?",
        "hi": "मुझे आपका सवाल सुनाई नहीं दिया। कृपया दोबारा बोलें।",
        "gu": "મને તમારો પ્રશ્ન સંભળાયો નહીં. કૃપા કરીને ફરીથી બોલો.",
    },
    ReasonCode.UNSAFE_INPUT: {
        "en": "I cannot help with that request.",
        "hi": "मैं इस अनुरोध में सहायता नहीं कर सकता।",
        "gu": "હું આ વિનંતીમાં મદદ કરી શકતો નથી.",
    },
    ReasonCode.PROMPT_INJECTION: {
        "en": "That request tries to change my instructions, so I will not act on it. Ask me about the knowledge base instead.",
        "hi": "यह अनुरोध मेरे निर्देश बदलने की कोशिश करता है, इसलिए मैं इसे नहीं मानूंगा।",
        "gu": "આ વિનંતી મારી સૂચનાઓ બદલવાનો પ્રયાસ કરે છે, તેથી હું તેને અનુસરીશ નહીં.",
    },
    ReasonCode.UNSUPPORTED_LANGUAGE: {
        "en": "I only cover English, Hindi and Gujarati right now.",
        "hi": "मैं फिलहाल केवल अंग्रेज़ी, हिंदी और गुजराती में उत्तर दे सकता हूँ।",
        "gu": "હું હાલમાં ફક્ત અંગ્રેજી, હિન્દી અને ગુજરાતીમાં જવાબ આપી શકું છું.",
    },
    ReasonCode.OUT_OF_CORPUS: {
        "en": "That is outside my knowledge base. I can only answer from the MSMARCO-XI corpus I was indexed on.",
        "hi": "यह मेरे ज्ञान भंडार से बाहर है। मैं केवल अनुक्रमित संग्रह से ही उत्तर दे सकता हूँ।",
        "gu": "તે મારા જ્ઞાન ભંડારની બહાર છે. હું ફક્ત અનુક્રમિત સંગ્રહમાંથી જ જવાબ આપી શકું છું.",
    },
    ReasonCode.LOW_CONFIDENCE: {
        "en": "I found something related but not close enough to answer confidently.",
        "hi": "मुझे कुछ मिला, पर आत्मविश्वास से उत्तर देने के लिए वह पर्याप्त नहीं है।",
        "gu": "મને કંઈક મળ્યું, પણ વિશ્વાસપૂર્વક જવાબ આપવા માટે તે પૂરતું નથી.",
    },
    ReasonCode.UNGROUNDED_OUTPUT: {
        "en": "I drafted an answer but could not verify it against the retrieved passages, so I am not going to state it.",
        "hi": "मैंने उत्तर तैयार किया, पर उसे प्राप्त अंशों से सत्यापित नहीं कर सका, इसलिए मैं इसे नहीं बताऊँगा।",
        "gu": "મેં જવાબ તૈયાર કર્યો, પણ પ્રાપ્ત ફકરાઓ સામે તેની ચકાસણી કરી શક્યો નહીં.",
    },
    ReasonCode.ANSWERED_FROM_GENERAL_KNOWLEDGE: {
        "en": "Not found in my knowledge base — answered from general knowledge instead, so treat it with more caution than a cited answer.",
        "hi": "यह मेरे ज्ञान भंडार में नहीं मिला — सामान्य जानकारी के आधार पर उत्तर दिया गया है, इसलिए इसे सावधानी से लें।",
        "gu": "આ મારા જ્ઞાન ભંડારમાં મળ્યું નથી — સામાન્ય જ્ઞાનના આધારે જવાબ આપ્યો છે, તેથી સાવધાની રાખો.",
    },
    ReasonCode.PROVIDER_UNAVAILABLE: {
        "en": "The answering model is unavailable, so I am returning the retrieved passage directly.",
        "hi": "उत्तर देने वाला मॉडल उपलब्ध नहीं है, इसलिए मैं संबंधित अंश सीधे दे रहा हूँ।",
        "gu": "જવાબ આપનાર મોડેલ ઉપલબ્ધ નથી, તેથી હું સંબંધિત ફકરો સીધો આપું છું.",
    },
}


def message(code: ReasonCode, lang: str | None = "en") -> str:
    slot = _MESSAGES.get(code, _MESSAGES[ReasonCode.OUT_OF_CORPUS])
    return slot.get(lang or "en", slot["en"])
