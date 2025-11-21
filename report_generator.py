"""
PDF report generator for DebateFin analysis results
使用 WeasyPrint 专业 PDF 库，完美支持中文
"""

from io import BytesIO
from datetime import datetime
from typing import Dict, Any
import json
import html

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


def generate_pdf_report(data: Dict[str, Any]) -> BytesIO:
    """
    Generate PDF report from analysis results using WeasyPrint (完美支持中文)
    
    Args:
        data: Dictionary containing:
            - analysis: Analysis results dictionary
            - debate_logs: Optional list of debate logs
            - ablation_results: Optional ablation study results
            - hallucination_checks: Optional hallucination check results
    
    Returns:
        BytesIO buffer containing PDF
    """
    if not WEASYPRINT_AVAILABLE:
        raise ImportError("weasyprint is required for PDF generation. Install with: pip install weasyprint")
    
    # Extract data
    analysis = data.get("analysis", data)  # Backward compatibility
    debate_logs = data.get("debate_logs", [])
    ablation_results = data.get("ablation_results")
    hallucination_checks = data.get("hallucination_checks", [])
    
    # Build HTML content
    html_content = build_html_report(analysis, debate_logs, ablation_results, hallucination_checks)
    
    # CSS styles (完美支持中文)
    css_content = """
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: "SimSun", "宋体", "Microsoft YaHei", "微软雅黑", "Arial", sans-serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #333;
    }
    h1 {
        font-size: 24pt;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    h2 {
        font-size: 16pt;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 12px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    h3 {
        font-size: 14pt;
        color: #34495e;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    th {
        background-color: #34495e;
        color: white;
        padding: 10px;
        text-align: center;
        font-weight: bold;
    }
    td {
        padding: 8px;
        border: 1px solid #ddd;
        text-align: left;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .metadata-table {
        background-color: #ecf0f1;
    }
    .warning {
        color: #e74c3c;
        font-style: italic;
    }
    .success {
        color: #27ae60;
        font-style: italic;
    }
    .footer {
        text-align: center;
        font-size: 8pt;
        color: #7f8c8d;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
    .page-break {
        page-break-before: always;
    }
    """
    
    # Generate PDF
    buffer = BytesIO()
    font_config = FontConfiguration()
    html_doc = HTML(string=html_content)
    css_doc = CSS(string=css_content, font_config=font_config)
    html_doc.write_pdf(buffer, stylesheets=[css_doc], font_config=font_config)
    buffer.seek(0)
    
    return buffer


def build_html_report(analysis: Dict, debate_logs: list, ablation_results: Dict, hallucination_checks: list) -> str:
    """构建 HTML 报告内容"""
    
    # Title
    html_parts = ['<html><head><meta charset="UTF-8"></head><body>']
    html_parts.append('<h1>DebateFin 金融分析报告</h1>')
    
    # Metadata
    ticker = html.escape(str(analysis.get("ticker", "N/A")))
    query = html.escape(str(analysis.get("query", "N/A")))
    timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    
    html_parts.append('<table class="metadata-table">')
    html_parts.append(f'<tr><th>股票代码</th><td>{ticker}</td></tr>')
    html_parts.append(f'<tr><th>分析查询</th><td>{query}</td></tr>')
    html_parts.append(f'<tr><th>生成时间</th><td>{timestamp}</td></tr>')
    html_parts.append('</table>')
    
    # Executive Summary
    html_parts.append('<h2>执行摘要</h2>')
    trader_prediction = analysis.get("trader_prediction", {})
    recommendation = html.escape(str(trader_prediction.get("recommendation", "持有")))
    html_parts.append(f'<p><strong>投资建议:</strong> {recommendation}</p>')
    
    # Final Synthesis
    final_synthesis = analysis.get("final_synthesis", "")
    if final_synthesis:
        html_parts.append('<h2>综合报告</h2>')
        html_parts.append(f'<p>{html.escape(final_synthesis).replace(chr(10), "<br/>")}</p>')
    
    # Analyst Evidence
    html_parts.append('<h2>分析师证据</h2>')
    analyst_evidence = analysis.get("analyst_evidence", {})
    if analyst_evidence:
        metrics = analyst_evidence.get("metrics", {})
        if metrics:
            html_parts.append('<table>')
            html_parts.append('<tr><th>指标</th><th>数值</th></tr>')
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html_parts.append(f'<tr><td>{html.escape(str(key))}</td><td>{value:.2f}</td></tr>')
            html_parts.append('</table>')
        
        analysis_text = analyst_evidence.get("analysis", "")
        if analysis_text:
            html_parts.append(f'<p>{html.escape(analysis_text[:1000]).replace(chr(10), "<br/>")}</p>')
    
    # Risk Flags
    html_parts.append('<h2>风险提示</h2>')
    risk_flags = analysis.get("risk_flags", [])
    if risk_flags:
        html_parts.append('<ul>')
        for flag in risk_flags:
            html_parts.append(f'<li>{html.escape(str(flag))}</li>')
        html_parts.append('</ul>')
    else:
        html_parts.append('<p>无重大风险标志</p>')
    
    # Debate Log
    debate_log = debate_logs or analysis.get("debate_log", [])
    if debate_log:
        html_parts.append('<div class="page-break"></div>')
        html_parts.append('<h2>辩论日志</h2>')
        for log_entry in debate_log:
            agent = html.escape(str(log_entry.get("agent", "Unknown")))
            action = html.escape(str(log_entry.get("action", "")))
            content = html.escape(str(log_entry.get("content", "")[:500]))
            round_num = log_entry.get("round", 0)
            
            html_parts.append(f'<h3>第 {round_num} 轮 - {agent} ({action})</h3>')
            html_parts.append(f'<p>{content.replace(chr(10), "<br/>")}</p>')
            
            # Hallucination check
            if "hallucination_check" in log_entry:
                h_check = log_entry["hallucination_check"]
                if isinstance(h_check, dict):
                    conf = h_check.get("confidence", 0)
                    if h_check.get("has_hallucination"):
                        html_parts.append(f'<p class="warning">⚠️ 幻觉检查: 检测到潜在问题 (置信度: {conf:.2f})</p>')
                    else:
                        html_parts.append(f'<p class="success">✅ 幻觉检查: 通过 (置信度: {conf:.2f})</p>')
    
    # Ablation Study Results
    if ablation_results:
        html_parts.append('<div class="page-break"></div>')
        html_parts.append('<h2>消融研究结果</h2>')
        
        metrics_with = ablation_results.get("with_debate", {}).get("metrics", {})
        metrics_without = ablation_results.get("without_debate", {}).get("metrics", {})
        
        if metrics_with and metrics_without:
            html_parts.append('<table>')
            html_parts.append('<tr><th>指标</th><th>有辩论</th><th>无辩论</th><th>差异</th></tr>')
            for metric in ["Sharpe", "MAE", "HallucinationConfidence", "QualityScore"]:
                val_with = metrics_with.get(metric, 0)
                val_without = metrics_without.get(metric, 0)
                diff = val_with - val_without
                html_parts.append(f'<tr><td>{metric}</td><td>{val_with:.3f}</td><td>{val_without:.3f}</td><td>{diff:.3f}</td></tr>')
            html_parts.append('</table>')
    
    # Hallucination Checks Summary
    if hallucination_checks:
        html_parts.append('<div class="page-break"></div>')
        html_parts.append('<h2>幻觉检查摘要</h2>')
        for check in hallucination_checks:
            agent = html.escape(str(check.get("agent", "Unknown")))
            round_num = check.get("round", 0)
            check_data = check.get("check", {})
            
            html_parts.append(f'<h3>{agent} (第{round_num}轮)</h3>')
            if check_data.get("has_hallucination"):
                issues = check_data.get("issues", [])
                issues_str = ', '.join([html.escape(str(i)) for i in issues])
                html_parts.append(f'<p class="warning">⚠️ 检测到潜在幻觉: {issues_str}</p>')
            else:
                conf = check_data.get("confidence", 0)
                html_parts.append(f'<p class="success">✅ 通过检查 (置信度: {conf:.2f})</p>')
    
    # Footer
    html_parts.append('<div class="footer">')
    html_parts.append('<p><em>本报告由DebateFin多智能体系统生成 | ICLR 2026 Workshop</em></p>')
    html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    return ''.join(html_parts)

