from pathlib import Path
from icartt_reader import ICARTTReader


ict_path = Path("C:\\Users\\mowma\\Downloads\\soot_DISCOVERAQ_staqs_20260127_da812cecb25fc115afc9aac311897b33\\DISCOVERAQ-PDS_P3B_20140802_R1_L2.ict")
r = ICARTTReader(ict_path)

df = r.read_table()              # DataFrame
meta = r.read_metadata()         # dict (best-effort)
vars_ = r.read_variable_defs()   # list of VariableDef (best-effort)
out_path = Path.home() / "Downloads" / "my_export.csv"
csv_path = r.to_csv(out=out_path)            # writes alongside the ICT



