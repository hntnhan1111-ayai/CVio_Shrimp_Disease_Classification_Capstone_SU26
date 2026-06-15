import { useEffect, useState } from "react";
import { Layout } from "./components/Layout";
import { WarningBox } from "./components/WarningBox";
import { loadCsv, loadText, type Row } from "./lib/csv";
import { edaTable, errorLog, yoloTable } from "./lib/paths";
import { ExecutiveSummary } from "./sections/ExecutiveSummary";
import { DatasetOverview } from "./sections/DatasetOverview";
import { LabelAudit } from "./sections/LabelAudit";
import { BoxStatistics } from "./sections/BoxStatistics";
import { GalleryReview } from "./sections/GalleryReview";
import { ModelBenchmark } from "./sections/ModelBenchmark";
import { ModelFailureAnalysis } from "./sections/ModelFailureAnalysis";
import { ResearchRecommendations } from "./sections/ResearchRecommendations";
type State={raw:Row[];annotated:Row[];boxes:Row[];mapping:Row[];errors:Row[];models:Row[];buckets:{size:string;rows:Row[]}[];logs:Record<string,string>};
const empty:State={raw:[],annotated:[],boxes:[],mapping:[],errors:[],models:[],buckets:[],logs:{}};
export default function App(){const [data,setData]=useState(empty);const [warnings,setWarnings]=useState<string[]>([]);const [loading,setLoading]=useState(true);
useEffect(()=>{const csv=async(path:string)=>{try{return await loadCsv(path)}catch(e){setWarnings(w=>[...w,`${path}: ${String(e)}`]);return[]}};const text=async(path:string)=>{try{return await loadText(path)}catch(e){setWarnings(w=>[...w,`${path}: ${String(e)}`]);return"Log unavailable."}};
Promise.all([csv(edaTable("summary_raw_class_counts.csv")),csv(edaTable("summary_annotated_counts.csv")),csv(edaTable("summary_box_class_counts.csv")),csv(edaTable("summary_source_by_canonical_class.csv")),csv(edaTable("label_errors.csv")),csv(yoloTable("final_model_comparison.csv")),csv(edaTable("box_size_buckets_at_640.csv")),csv(edaTable("box_size_buckets_at_1024.csv")),csv(edaTable("box_size_buckets_at_1280.csv")),text(errorLog("yolov13n_train_error.txt")),text(errorLog("yolov13s_train_error.txt"))]).then(([raw,annotated,boxes,mapping,errors,models,b640,b1024,b1280,nlog,slog])=>setData({raw,annotated,boxes,mapping,errors,models,buckets:[{size:"640",rows:b640},{size:"1024",rows:b1024},{size:"1280",rows:b1280}],logs:{"YOLOv13n loading error":nlog as string,"YOLOv13s loading error":slog as string}})).finally(()=>setLoading(false));},[]);
return <Layout>{loading?<div className="loading">Loading report artifacts…</div>:<>{warnings.length>0&&<WarningBox>Some optional artifacts could not be loaded:<ul>{warnings.map(w=><li key={w}>{w}</li>)}</ul></WarningBox>}<ExecutiveSummary annotated={data.annotated} boxes={data.boxes} models={data.models}/><DatasetOverview raw={data.raw} annotated={data.annotated} boxes={data.boxes}/><LabelAudit mapping={data.mapping} errors={data.errors}/><BoxStatistics buckets={data.buckets}/><GalleryReview/><ModelBenchmark models={data.models}/><ModelFailureAnalysis logs={data.logs}/><ResearchRecommendations/></>}</Layout>}
