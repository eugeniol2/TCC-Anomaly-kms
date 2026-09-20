
#     python -m src.examples.reproducibility


from numpy.random import default_rng

SEMENTE = 1

def endereco(gerador):

    return str(gerador.bit_generator.seed_seq.spawn_key)


# A raiz ---------------------------------------------------------------

rng = default_rng(SEMENTE)

print("raiz, da semente", SEMENTE)
print(f"   endereco {endereco(rng):<8} valor {rng.integers(100, 1000)}")


# As filhas, geradas a partir da raiz -----------------------------------

filha_a, filha_b = rng.spawn(2)

print()
print("   duas filhas da raiz")
print(f"      filha A   endereco {endereco(filha_a):<8} valor {filha_a.integers(100, 1000)}")
print(f"      filha B   endereco {endereco(filha_b):<8} valor {filha_b.integers(100, 1000)}")


# As netas, geradas a partir da filha A ---------------------------------

neta_a, neta_b = filha_a.spawn(2)

print()
print("      duas netas, geradas pela filha A")
print(f"         neta A   endereco {endereco(neta_a):<8} valor {neta_a.integers(100, 1000)}")
print(f"         neta B   endereco {endereco(neta_b):<8} valor {neta_b.integers(100, 1000)}")


print()
print("Rode de novo: os cinco numeros se repetem, na mesma ordem.")


# As netas, geradas a partir da filha B ---------------------------------

neta_a, neta_b = filha_b.spawn(2)

print()
print("      duas netas, geradas pela filha B")
print(f"         neta A   endereco {endereco(neta_a):<8} valor {neta_a.integers(100, 1000)}")
print(f"         neta B   endereco {endereco(neta_b):<8} valor {neta_b.integers(100, 1000)}")


print()
print("Rode de novo: os cinco numeros se repetem, na mesma ordem.")