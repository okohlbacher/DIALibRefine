#!/usr/bin/env python3
"""A synthetic single-run DIA-NN report.parquet for the end-to-end tests.

Peptides come from a tryptic digest of a FASTA; RT and 1/K0 are deterministic,
LEARNABLE functions of the sequence that the stock models do not know exactly
(hydrophobicity- and mass-based), so a fine-tune must move the held-out metric
in the right direction, and a broken one cannot pass by chance. Nothing here is
a physical model of anything; it is a fixture.

  synth_report.py <proteins.fasta> <out.parquet> [--precursors 900] [--seed 1]
"""
import argparse, re, sys, zlib
import numpy as np
try:
    import pyarrow as pa, pyarrow.parquet as pq
except ImportError:
    sys.exit("SKIP: pyarrow is not available")

ap = argparse.ArgumentParser()
ap.add_argument("fasta", help="a protein FASTA, or - for synthetic proteins"); ap.add_argument("out")
ap.add_argument("--precursors", type=int, default=1600)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

MONO = {"G":57.02146,"A":71.03711,"S":87.03203,"P":97.05276,"V":99.06841,"T":101.04768,"C":160.03065,"L":113.08406,"I":113.08406,
        "N":114.04293,"D":115.02694,"Q":128.05858,"K":128.09496,"E":129.04259,"M":131.04049,"H":137.05891,"F":147.06841,"R":156.10111,
        "Y":163.06333,"W":186.07931}
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
      "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}

def digest(prots):
    rows = []
    for name, seq in prots.items():
        for pep in re.split(r"(?<=[KR])(?!P)", seq):
            if 7 <= len(pep) <= 30 and set(pep) <= set(MONO):
                rows.append((name, pep))
    return rows

prots = {}
if a.fasta not in ("-", ""):
    name = None
    for line in open(a.fasta):
        line = line.strip()
        if line.startswith(">"): name = line[1:].split()[0]; prots[name] = []
        elif name: prots[name].append(line)
    prots = {k: "".join(v) for k, v in prots.items()}
rows = digest(prots)
if len(rows) < 2 * a.precursors:
    # Not enough real peptides (DIALibGen's example FASTA digests to 24):
    # add synthetic proteins drawn from UniProt's background amino-acid
    # frequencies, deterministic under the seed. They are fixtures, not biology.
    aa = "ACDEFGHIKLMNPQRSTVWY"
    freq = np.array([8.25,1.37,5.45,6.75,3.86,7.07,2.27,5.96,5.84,9.66,2.42,4.06,4.70,3.93,5.53,6.56,5.34,6.87,1.08,2.92]); freq /= freq.sum()
    i = 0
    while len(rows) < 2 * a.precursors:
        prots[f"SYN{i:04d}"] = "".join(rng.choice(list(aa), size=320, p=freq)); i += 1
        rows = digest(prots)
rng.shuffle(rows)

def mass(pep): return sum(MONO[c] for c in pep) + 18.01056
def rt_minutes(pep):  # hydrophobicity + length, mapped to a 30-min gradient, plus small noise
    h = sum(KD[c] for c in pep) / len(pep)
    return float(np.clip(10.5 + 4.0 * h + 0.25 * len(pep) + rng.normal(0, 0.15), 0.5, 29.5))
# A mass-and-charge law near what the stock model already knows, plus a
# composition-linear term it does not (a seeded per-residue table): the part
# a fine-tune has to learn, and can.
RES_CCS = {c: float(v) for c, v in zip(MONO, rng.normal(0, 6.0, size=len(MONO)))}
def ccs(pep, z):
    return 5.2 * mass(pep) ** 0.60 * (1.0 + 0.10 * (z - 2)) + sum(RES_CCS[c] for c in pep)
def mobility(ccs_, mz, z):  # Mason-Schamp with ODIA's constants (Library.h)
    m_ion = mz * z; mu = m_ion * 28.0134 / (m_ion + 28.0134)
    return ccs_ * np.sqrt(mu * 305.0) / (18509.0 * z)

out = {k: [] for k in ("Run","Protein.Group","Modified.Sequence","Stripped.Sequence","Precursor.Charge","Precursor.Mz","RT","IM","Q.Value","Decoy")}
n = 0
for name, pep in rows:
    rt = rt_minutes(pep)
    for z in (2, 3) if len(pep) > 12 else (2,):
        mz = (mass(pep) + z * 1.007276) / z
        modseq = pep.replace("C", "C(UniMod:4)")
        out["Run"].append("synthetic"); out["Protein.Group"].append(name)
        out["Modified.Sequence"].append(modseq); out["Stripped.Sequence"].append(pep)
        out["Precursor.Charge"].append(z); out["Precursor.Mz"].append(mz)
        out["RT"].append(rt + rng.normal(0, 0.03)); out["IM"].append(mobility(ccs(pep, z), mz, z) * (1 + rng.normal(0, 0.004)))
        out["Q.Value"].append(0.001); out["Decoy"].append(0)
        n += 1
        if n >= a.precursors: break
    if n >= a.precursors: break
t = pa.table({k: pa.array(v) for k, v in out.items()})
pq.write_table(t, a.out)
test = sum(1 for g in set(out["Protein.Group"]) if zlib.crc32(g.encode()) % 5 == 0)
print(f"wrote {a.out}: {n} precursors, {len(set(out['Protein.Group']))} protein groups ({test} in the TEST cohort)")
