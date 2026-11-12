import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const base = path.resolve(process.env.FINAI_TABLES_OUT ?? 'analysis/publication_tables_20261112');
const data = JSON.parse(await fs.readFile(path.join(base, 'workbook_data.json'), 'utf8'));
const audits = JSON.parse(await fs.readFile(path.join(base, 'audit.json'), 'utf8'));
const out = path.resolve(process.env.FINAI_WORKBOOK_OUT ?? 'outputs/publication_tables_20261112');
await fs.mkdir(out, {recursive:true});
const names = {ln_at:'Log(total assets)',rd_at:'R&D / assets',rd_missing:'R&D missing indicator',
 capx_at:'Capex / assets',tobinq:"Tobin's q",roa:'Operating ROA',tangibility:'Tangibility',
 cash_at:'Cash / assets',leverage:'Book leverage',sales_growth:'Sales growth',
 ln_item1:'Log(Item 1 sentences)',ln_document:'Log(10-K sentences)'};
const specs = [1,2,3,4,6];
const notes = [
 ['Status','Draft tables for supervisor review. Source templates and original data are unchanged.'],
 ['Sample','Broad filing sample and professor price-screened sample are separate workbooks. Both use filings dated 2015-2025 and reporting periods ending in 2015-2025.'],
 ['Professor sample','A filing is eligible if used by at least one professor-panel stock-month with price >= $5 and non-missing raw ExpertiseRatio and AIRatio. This is a reconstruction, not confirmation of all final return-test transformations.'],
 ['Filing unit','One CIK, filing date and reporting-period end key. For company averages, one latest filing per CIK and reporting period. Share classes do not add filing observations.'],
 ['Excerpts','Panel A ranks filing-weighted mean expertise (top 10); panel B ranks mean AI (top 5) among CIKs with zero expertise in every included filing. At least two filings.'],
 ['Excerpt selection','Use the filing with the maximum relevant ratio, latest period on ties. Select its highest-topic-weight eligible Item 1 sentence, at least 10 words. A requires expertise; B excludes noise. Quotes are verbatim. Missing eligible quotes are flagged, not replaced by a different firm.'],
 ['Length sensitivity','The separate excerpt sheet repeats the ranking after requiring at least 25 Item 1 sentences. This optional screen is not imposed on the main ranking.'],
 ['Reporting year','Excerpt year is the calendar year of reporting-period end, not necessarily the Compustat fiscal-year label.'],
 ['Accounting alignment','Use gvkey and exact accounting period end where available. Otherwise use the Jan-May fiscal-year convention only when raw fiscal-end month matches. See audit for match counts. This is not a monthly availability-lag merge.'],
 ['Accounting source','USD, standardized, consolidated INDL records in the detailed accounting download; positive assets required. Missing items are not filled with zero except R&D in specified columns.'],
 ['Variable definitions','Assets log=ln(at); R&D/assets=xrd/at; capex/assets=capx/at; q=(at-ceq+prcc_f*csho)/at; operating ROA=oibdp/at; tangibility=ppent/at; cash=che/at; leverage=(dltt+dlc)/at.'],
 ['Sales growth','sale(t)/sale(t-1)-1 for consecutive Compustat fiscal years with positive prior sales.'],
 ['Regression period','Accounting fiscal years 2015-2025. Outcome percentages equal raw Item 1 ratios times 100. Regressors are not standardized.'],
 ['Winsorization','Each continuous outcome and regressor is clipped at within-fiscal-year 1st/99th percentiles. R&D imputation follows winsorization; dummies are not winsorized. Exact bounds are in the CSV outputs.'],
 ['Specifications','Columns 1-4 have FF48 and fiscal-year effects. Column 6 has PERMCO and fiscal-year effects. 1: missing R&D=0; 2 and 6: same plus missing dummy; 3: reported R&D only; 4: R&D omitted.'],
 ['Inference','OLS with absorbed effects; SE clustered by PERMCO. Full fixed-effect rank enters finite-sample and adjusted-R2 degrees of freedom. Two-sided p-values use cluster count minus one degrees of freedom. Singleton observations are retained; this can differ from reghdfe defaults.'],
 ['Significance','*** p<0.01; ** p<0.05; * p<0.10. Parentheses report t-statistics, not standard errors.'],
 ['Firm age missing','The extract starts in 2010 and cannot establish first Compustat appearance. Firm age is omitted in all columns, so these are not final exact reproductions of Table 3.3.1.'],
 ['Length definition','10-K sentence count is used as the template-permitted alternative to submission-file byte size. Item 1 length uses sentence count.'],
 ['Reduced specification','Column 5 (and its AIRatio counterpart 11) is not estimated because the reduced control set is not agreed. No controls were selected using significance.'],
 ['Ratio reconciliation','Source maxima differ from manuscript summary maxima. Raw panel values are retained for excerpt ranking. Regression winsorization is explicit and does not establish the professor used the same transformation.'],
 ['Earnings calls','Call-level files were not located; Table 3.4.1 is omitted, not filled from old pooled correlation summaries.'],
 ['Patents','Table 3.3.2 is not regenerated in this workbook while the patent classification and coverage work remain pending.']
];

function style(sheet, rows, cols) {
 sheet.showGridLines=false;
 const all=sheet.getRangeByIndexes(0,0,Math.max(rows,5),cols);
 all.format.font={name:'Arial',size:10,color:'#202124'};
 all.format.rowHeight=21;
 all.format.verticalAlignment='center';
 sheet.getRangeByIndexes(1,0,1,cols).format.font={name:'Arial',size:14,bold:true};
}
function header(sheet, row, labels) {
 const r=sheet.getRangeByIndexes(row,0,1,labels.length); r.values=[labels];
 r.format.fill='#243B35'; r.format.font={name:'Arial',size:10,color:'#FFFFFF',bold:true};
 r.format.wrapText=true; r.format.rowHeight=32;
}
function rawSheet(wb, title, rows) {
 const s=wb.worksheets.getItem(title);
 const cols=Object.keys(rows[0]??{value:''});
 style(s,rows.length+1,cols.length);
 header(s,0,cols);
 if(rows.length) s.getRangeByIndexes(1,0,rows.length,cols.length).values=rows.map(r=>cols.map(k=>r[k]??null));
 s.getRangeByIndexes(0,0,rows.length+1,cols.length).format.columnWidth=18;
 s.freezePanes.freezeRows(1);
 return s;
}
function examples(wb, title, rows, sample) {
 const s=wb.worksheets.getItem(title);
 style(s,rows.length+6,8);
 s.getRange('A2').values=[['Table 3.1.1: '+(sample==='broad'?'Broad sample':'Professor sample')]];
 s.getRange('A3').values=[['DRAFT: flagged missing quotes retained. Reporting year is period-end year. Ratios shown in percent.']];
 s.getRange('A3:H3').format.font={name:'Arial',size:10,italic:true,color:'#555555'};
 header(s,4,['Panel / rank','Company','Reporting year','Filings','Avg expertise (%)','Avg AI (%)','Verbatim excerpt','Status']);
 const values=rows.map(r=>[r.panel+' '+r.rank,r.company,r.fiscal_year_label,r.filings,
   r.average_expertise_pct,r.average_ai_pct,r.excerpt||'No eligible non-noise sentence in selected filing',r.excerpt_status]);
 s.getRangeByIndexes(5,0,rows.length,8).values=values;
 [12,30,16,10,19,16,95,25].forEach((v,i)=>s.getRangeByIndexes(0,i,rows.length+6,1).format.columnWidth=v);
 s.getRangeByIndexes(5,4,rows.length,2).setNumberFormat('0.000');
 s.getRangeByIndexes(5,0,rows.length,8).format.wrapText=true;
 for(let i=0;i<rows.length;i++) {
  const count=Math.ceil((values[i][6].length||1)/105);
  s.getRangeByIndexes(i+5,0,1,8).format.rowHeight=Math.max(48, count*15+12);
  s.getRangeByIndexes(i+5,0,1,8).format.verticalAlignment='top';
  if(rows[i].excerpt_status!=='selected') s.getRangeByIndexes(i+5,6,1,2).format.fill='#FFF0D4';
 }
 s.freezePanes.freezeRows(5);
 return s;
}
function regression(wb, name, dependent, input, sample) {
 const s=wb.worksheets.getItem(name);
 style(s,48,6);
 s.getRange('A2').values=[['Table 3.3.1: '+(dependent==='expertise_pct'?'ExpertiseRatio':'AIRatio')]];
 s.getRange('A3').values=[['DRAFT: firm age omitted; 10-K length measured in sentences. Outcomes in percentage points.']];
 header(s,4,['Characteristic',...specs.map(n=>'('+n+')')]);
 s.getRange('A1:A48').format.columnWidth=31;
 s.getRange('B1:F48').format.columnWidth=17;
 let row=5;
 for(const [key,label] of Object.entries(names)) {
  s.getRangeByIndexes(row,0,1,1).values=[[label]];
  for(let j=0;j<specs.length;j++) {
   const index=input.coefficients.findIndex(r=>r.dependent===dependent&&r.specification===specs[j]&&r.variable===key);
   if(index<0) continue;
   const excelRow=index+2;
   s.getRangeByIndexes(row,j+1,1,1).formulas=[[
    '=TEXT(Estimates!B'+excelRow+',"0.000")&IF(Estimates!E'+excelRow+'<0.01,"***",IF(Estimates!E'+excelRow+'<0.05,"**",IF(Estimates!E'+excelRow+'<0.1,"*","")))'
   ]];
   s.getRangeByIndexes(row+1,j+1,1,1).formulas=[['="("&TEXT(Estimates!D'+excelRow+',"0.00")&")"']];
  }
  s.getRangeByIndexes(row+1,0,1,6).format.font={name:'Arial',size:10,color:'#666666'};
  row+=2;
 }
 row++;
 for(const [key,label] of [['n','Observations'],['firms','Firms'],['adjusted_r2','Adjusted R2']]) {
  s.getRangeByIndexes(row,0,1,1).values=[[label]];
  s.getRangeByIndexes(row,1,1,5).values=[specs.map(n=>input.models.find(r=>r.dependent===dependent&&r.specification===n)?.[key]??null)];
  s.getRangeByIndexes(row,1,1,5).setNumberFormat(key==='adjusted_r2'?'0.000':'#,##0');
  row++;
 }
 s.getRangeByIndexes(row+1,0,1,1).values=[['*** p<0.01; ** p<0.05; * p<0.10. Firm-clustered t-statistics in parentheses.']];
 s.getRangeByIndexes(row+2,0,1,1).values=[['Columns 1-4: FF48 and year FE. Column 6: firm and year FE. See Definitions for missing controls and sample details.']];
 s.freezePanes.freezeRows(5);
 return s;
}

for(const [sample,input] of Object.entries(data)) {
 const wb=Workbook.create();
 ['Excerpts','Expertise','AI ratio','Excerpts length 25','Control correlations','Control VIF','Estimates','Models','Sample audit','Definitions']
  .forEach(name=>wb.worksheets.add(name));
 const ex=examples(wb,'Excerpts',input.examples,sample);
 const expertise=regression(wb,'Expertise', 'expertise_pct',input,sample);
 regression(wb,'AI ratio','ai_pct',input,sample);
 examples(wb,'Excerpts length 25',input.examples_length25,sample);
 // Keep numeric estimate columns stable for the displayed coefficient formulas.
 const ordered=input.coefficients.map(r=>({variable:r.variable,coefficient:r.coefficient,se:r.se,t:r.t,p:r.p,
   dependent:r.dependent,specification:r.specification}));
 rawSheet(wb,'Estimates',ordered);
 rawSheet(wb,'Models',input.models);
 if(input.vif?.length) rawSheet(wb,'Control VIF',input.vif);
 if(input.correlations?.length) {
  const s=wb.worksheets.getItem('Control correlations');
  const vars=input.correlation_variables;
  style(s,vars.length+1,vars.length+1);
  header(s,0,['Variable',...vars]);
  s.getRangeByIndexes(1,0,vars.length,vars.length+1).values=vars.map((v,i)=>[v,...input.correlations[i]]);
  s.getRangeByIndexes(0,0,vars.length+1,vars.length+1).format.columnWidth=14;
  s.getRangeByIndexes(1,1,vars.length,vars.length).setNumberFormat('0.000');
  s.freezePanes.freezeRows(1);
 }
 const auditRows=Object.entries(audits[sample]).filter(([k])=>k!=='model_statistics').map(([key,value])=>({
  measure:key,value:typeof value==='object'?JSON.stringify(value):value}));
 rawSheet(wb,'Sample audit',auditRows);
 const def=wb.worksheets.getItem('Definitions');
 style(def,notes.length+1,2);header(def,0,['Topic','Definition or limitation']);
 def.getRangeByIndexes(1,0,notes.length,2).values=notes;
 def.getRange('A1:A30').format.columnWidth=26;
 def.getRange('B1:B30').format.columnWidth=120;
 def.getRangeByIndexes(1,0,notes.length,2).format.wrapText=true;
 def.getRangeByIndexes(1,0,notes.length,2).format.rowHeight=48;
 def.freezePanes.freezeRows(1);
 wb.recalculate();
 console.log((await wb.inspect({kind:'region',sheetId:'Expertise',range:'A5:F12',maxChars:1800,tableMaxRows:8,tableMaxCols:6})).ndjson);
 for(const [sheetName,range,suffix] of [['Expertise','A1:F39','regression'],['Excerpts','A1:H8','examples']]) {
  const preview=await wb.render({sheetName,autoCrop:'all',scale:1,format:'png'});
  await fs.writeFile(path.join(out,sample+'_'+suffix+'.png'),new Uint8Array(await preview.arrayBuffer()));
 }
 const file=await SpreadsheetFile.exportXlsx(wb);
 await file.save(path.join(out,sample+'_publication_tables_DRAFT.xlsx'));
}
