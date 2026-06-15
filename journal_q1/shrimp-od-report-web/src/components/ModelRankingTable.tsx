import type { Row } from "../lib/csv";
import { DataTable } from "./DataTable";
export function ModelRankingTable({rows}:{rows:Row[]}) { return <DataTable title="Model ranking (searchable)" rows={rows} columns={["model","status","map50","map50_95","diagnosis_macro_f1_disease_only","disease_miss_rate_no_det","fps_total","model_size_mb","params_m","mobile_edge_score"]}/>; }
