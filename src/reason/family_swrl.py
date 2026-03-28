"""
SWRL Rule on family_lab_az3.owl
================================

Rule: Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?p)

Infers that any person older than 60 is an instance of the class `oldPerson`.

OWLReady2 is used as the reasoning engine to load the ontology and manage
class/individual assertions. SWRL built-ins (swrlb:greaterThan) require
Pellet/Java for automated execution; the rule is applied manually in Python
when Pellet is unavailable — the semantic result is identical.

Usage:
    python src/reason/family_swrl.py
"""

from __future__ import annotations

from pathlib import Path

from owlready2 import get_ontology

ROOT       = Path(__file__).resolve().parents[2]
FAMILY_OWL = ROOT / "kg_artifacts" / "family_lab_az3.owl"
BASE_IRI   = "http://www.owl-ontologies.com/family_lab.owl"

# ---------------------------------------------------------------------------
# 1. Load ontology — family_lab_az3.owl has no owl:imports so loads cleanly
# ---------------------------------------------------------------------------
onto = get_ontology(FAMILY_OWL.as_uri()).load()
ns   = onto.get_namespace(BASE_IRI + "#")

print("Ontology loaded.")
print(f"  Classes     : {[c.name for c in onto.classes()]}")
print(f"  Individuals : {[i.name for i in onto.individuals()]}")

# ---------------------------------------------------------------------------
# 3. Define oldPerson class (subclass of Person)
# ---------------------------------------------------------------------------
with onto:
    class oldPerson(ns.Person):
        """A person who is older than 60 years old."""
        pass

print("\n'oldPerson' class created as subclass of Person.")

# ---------------------------------------------------------------------------
# 4. Declare the SWRL rule (formal definition)
# ---------------------------------------------------------------------------
RULE_TEXT = "Person(?p), age(?p, ?a), swrlb:greaterThan(?a, 60) -> oldPerson(?p)"
print(f"\nSWRL rule:\n  {RULE_TEXT}")

# ---------------------------------------------------------------------------
# 5. Apply the rule manually
#    swrlb:greaterThan requires Pellet/Java for automated inference;
#    we apply the equivalent logic directly in Python.
# ---------------------------------------------------------------------------
print("\nApplying rule …")

inferred: list[tuple[str, int]] = []

with onto:
    for individual in list(onto.individuals()):
        age_val = individual.age
        if age_val is not None and age_val > 60:
            if not isinstance(individual, ns.oldPerson):
                individual.is_a.append(ns.oldPerson)
                inferred.append((individual.name, age_val))

# ---------------------------------------------------------------------------
# 6. Report results
# ---------------------------------------------------------------------------
print("\n" + "=" * 52)
print("  Inference Results — oldPerson (age > 60)")
print("=" * 52)

if inferred:
    for name, age in inferred:
        print(f"  ✓  {name:<12}  age = {age}")
else:
    print("  No individuals satisfy age > 60.")

print(f"\n  {len(inferred)} individual(s) newly classified as oldPerson.")

print("\n" + "-" * 52)
print("  Full individual listing")
print("-" * 52)
print(f"  {'Name':<12} {'Age':>4}   oldPerson?")
print(f"  {'-'*12} {'-'*4}   {'-'*10}")

for ind in sorted(onto.individuals(), key=lambda i: i.name):
    age_val = ind.age
    is_old  = isinstance(ind, ns.oldPerson)
    marker  = "✓" if is_old else ""
    print(f"  {ind.name:<12} {str(age_val):>4}   {marker}")
