"""Canonical student-level, immigration, program, and campus vocabulary.

Terms are evidence of language used, NOT proof of the speaker's own identity.
"""

CONTEXT_PATTERNS = {
    "student_level": {
        "undergraduate": (
            "undergraduate student", "undergrad student", "undergraduate",
            "undergrad", "bachelor's student", "bachelor student",
            "bachelors student", "ug student", "freshman", "sophomore",
            "junior student", "senior student",
            # bare "junior"/"senior" omitted: "junior developer", "senior citizen".
        ),
        "graduate": (
            "graduate student", "grad student", "master's student",
            "masters student", "ms student", "m.s. student", "phd student",
            "phd candidate", "doctoral student", "doctoral candidate",
            # bare "graduate" omitted on purpose: "I'm a graduate" may be an alumnus.
        ),
    },
    "visa_status": {
        "F-1": ("f-1", "f-1 visa", "f-1 student"),   # also matches f1 / f 1
        "J-1": ("j-1", "j-1 visa", "j-1 student"),
    },
    "international_status": {"international": ("international student",)},
    "program": {
        "coterminal": ("coterminal", "co-terminal"),
        # An MBA is also a completed credential. Remember a program only when
        # the speaker claims current student/program participation.
        "MBA": ("mba student", "mba program"),
        "law": ("law student", "chicago-kent student"),
    },
    "campus": {
        "Mies": ("mies campus", "main campus"),
        "Downtown": ("downtown campus",),
        "Rice": ("rice campus",),  # Rice Campus is not the Downtown Campus.
    },
}

LEVEL_MENTIONS = {
    "undergraduate": (
        "undergraduate", "undergraduates", "undergrad", "undergrads", "ug",
        "bachelor", "bachelor's", "bachelors", "freshman", "sophomore",
    ),
    "graduate": (
        "graduate", "grad", "grads", "master's", "masters", "master of", "ms",
        "m.s.", "m.s", "phd", "ph.d.", "ph.d", "doctoral", "doctorate",
        "postgraduate", "graduate program",
    ),
}

IMMIGRATION_TERMS = (
    "i-20", "i20", "cpt", "f-1", "f1", "j-1", "j1", "sevis", "sevp", "dso",
    "i-94", "work authorization", "employment authorization", "h-1b", "h1b",
    "reduced course load", "travel signature", "stem opt",
)
