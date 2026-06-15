export const DATA = "/report-data";
export const edaTable = (name: string) => `${DATA}/eda/tables/${name}`;
export const edaFigure = (name: string) => `${DATA}/eda/figures/${name}`;
export const gallery = (name: string) => `${DATA}/eda/galleries/${name}`;
export const yoloTable = (name: string) => `${DATA}/yolo_results/tables/${name}`;
export const errorLog = (name: string) => `${DATA}/yolo_results/errors/${name}`;
export const runRoot = `${DATA}/yolo_results/runs_train_30ep/shrimp_od_yolo_ultralytics_under12M_17models_rtx4090_v3_FIXED_OD_ONLY/runs_train_30ep`;
export const runModels = ["yolov8n","yolov5nu","yolov5su","yolov8s","yolov5n6u","yolo11n","yolov9s","yolo26n","yolo12n","yolov10n","yolov9t","yolo11s","yolov10s","yolo12s","yolo26s"];
export const runPath = (model: string, file: string) => `${runRoot}/${model}_shrimpOD_2cls_img1024_ep30_seed42_diseaseOnly/${file}`;
