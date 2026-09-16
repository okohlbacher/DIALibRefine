// Copyright (c) 2026, Oliver Kohlbacher and the DIALibRefine authors.
// SPDX-License-Identifier: BSD-3-Clause
//
// The join key. This is the test that matters most in the whole tool: DIA-NN
// writes C(UniMod:4) and an OpenMS-generated library writes C(Carbamidomethyl),
// and on an alkylated sample a verbatim join drops every cysteine precursor --
// roughly 10% of the identifications -- with no error at all.
#include <odia/LibraryRefiner.h>
#include <iostream>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string& what)
{
  if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}

int main()
{
  using ODIA::canonicalModifiedSequence;
  using ODIA::LibraryRefiner;

  check(canonicalModifiedSequence("AC(UniMod:4)DEK") == canonicalModifiedSequence("AC(Carbamidomethyl)DEK"),
        "UniMod:4 and Carbamidomethyl must canonicalise to the same string");
  check(LibraryRefiner::key("AC(UniMod:4)DEK", 2) == LibraryRefiner::key("AC(Carbamidomethyl)DEK", 2),
        "the join key must be alkylation-naming independent");
  check(LibraryRefiner::key("AC(UniMod:4)DEK", 2) != LibraryRefiner::key("AC(UniMod:4)DEK", 3),
        "charge must be part of the key");
  check(canonicalModifiedSequence("AM(Oxidation)K") == canonicalModifiedSequence("AM(UniMod:35)K"),
        "oxidation aliases");
  check(canonicalModifiedSequence("PEPTIDEK") == "PEPTIDEK", "unmodified passes through");
  check(canonicalModifiedSequence("AC[Carbamidomethyl]DEK") == canonicalModifiedSequence("AC(UniMod:4)DEK"),
        "square and round bracket dialects agree");

  // A bare mass shift has no accession. It must NOT collide with a named
  // modification, because a wrong collision is a silently wrong join.
  check(canonicalModifiedSequence("AC(+57.0215)DEK") != canonicalModifiedSequence("AC(UniMod:4)DEK"),
        "an unnamed mass shift must not be assumed to be carbamidomethyl");

  // Nested brackets are one token (review M12): the first implementation stopped
  // at the first ')' and K(Label:13C(6)15N(2)) could never match its accession.
  check(canonicalModifiedSequence("PEPTIDEK(Label:13C(6)15N(2))") == canonicalModifiedSequence("PEPTIDEK(UniMod:259)"),
        "nested isotope-label token must canonicalise whole");
  check(canonicalModifiedSequence("PEPTIDEK(Label:13C(6)15N(2))") == "PEPTIDEK(UniMod:259)",
        "and to the accession form");

  // Unknown tokens are counted, known ones are not.
  std::size_t unknown = 0;
  canonicalModifiedSequence("AC(Carbamidomethyl)DM(Oxidation)K(Foo)R", &unknown);
  check(unknown == 1, "exactly the unknown token is counted");

  if (failures == 0) { std::cout << "canonical_modseq: all checks passed\n"; }
  return failures == 0 ? 0 : 1;
}
