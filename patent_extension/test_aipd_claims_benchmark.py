import tempfile
import unittest
import zipfile
from pathlib import Path

from aipd_claims_benchmark import claims_for


class ClaimExtractionTests(unittest.TestCase):
    def test_filters_by_publication_and_restores_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claims.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "claims.tsv", "pgpub_id\tclaim_sequence\tclaim_text\n"
                    "20230000001\t2\tSecond claim\n"
                    "20230000002\t1\tOther patent\n"
                    "20230000001\t1\tFirst claim\n"
                    "20230000001\t3\t\n",
                )
            result = claims_for(path, {"20230000001"})
            self.assertEqual(list(result.pgpub_id), ["20230000001"])
            self.assertEqual(result.claim_text.iloc[0], "First claim Second claim")
            self.assertEqual(result.claim_count.iloc[0], 2)

    def test_conflicting_claim_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claims.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "claims.tsv", "pgpub_id\tclaim_sequence\tclaim_text\n"
                    "20230000001\t1\tFirst version\n"
                    "20230000001\t1\tOther version\n",
                )
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                claims_for(path, {"20230000001"})


if __name__ == "__main__":
    unittest.main()
