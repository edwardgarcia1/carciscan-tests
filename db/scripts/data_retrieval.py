import duckdb
import time

def get_smiles(name):
    start_time = time.perf_counter()
    con = duckdb.connect(database='../carciscan.db')
    found_synonym = con.execute("""
        SET default_collation = 'nocase';
        SELECT CID, jaro_winkler_similarity(LOWER(Synonyms), LOWER($Name)) AS score, Synonyms
        FROM synonyms
        WHERE score > 0.90
        ORDER BY score DESC
    """,{
        "Name": name
    }).fetchone()
    cid, search_score, synonym = found_synonym
    if cid:

        found_smiles = con.execute("""
            SELECT SMILES
            FROM smiles
            WHERE CID = $CID
            LIMIT 1
        """,{
            "CID": cid
        }).fetchone()

        con.close()
        smiles = found_smiles[0]
        end_time = time.perf_counter()
        print(f"Took {round(end_time - start_time, 2)}s")
        return smiles, cid
    else:
        con.close()
        print(f"[NOT FOUND] No Synonym: {name}")
        return None

def get_annotations(cid):
    con = duckdb.connect(database='../carciscan.db')
    annotations = con.execute("""
        SELECT Categories, "Route of Exposure", "Mechanism of Toxicity", Metabolism, "Lethal Dose",
            Carcinogenicity, "Uses/Sources", "Minimum Risk Level", "Health Effects", Symptoms, Treatment
        FROM t3db
        WHERE CID = $cid
        LIMIT 1
    """,{
        "cid": cid
    }).fetchone()
    con.close()

    if annotations:
        print(annotations)
        return annotations
    else:
        print(f"[NOT FOUND] No T3DB annotations: CID {cid}")
        return None

smiles, cid = get_smiles('salt')
print(f"CID: {cid} \nSMILES: {smiles}")
get_annotations(cid)