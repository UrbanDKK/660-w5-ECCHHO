import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenAI · ECCHHO Media",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: #0a0a0a;
    border-radius: 14px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border-left: 5px solid #10b981;
}
.hero-tag  { font-size:0.72rem; color:#10b981; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.6rem; }
.hero-h1   { font-size:2rem; font-weight:700; color:#ffffff; line-height:1.25; margin:0 0 0.5rem; }
.hero-sub  { font-size:0.95rem; color:#9ca3af; margin:0; }
.hero-team { font-size:0.78rem; color:#4b5563; margin-top:1rem; }

.kpi { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:1.1rem 1.3rem; text-align:center; }
.kpi-label { font-size:0.72rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:2rem; font-weight:700; color:#111827; line-height:1.2; margin:0.2rem 0; }
.kpi-delta { font-size:0.78rem; color:#6b7280; }
.kpi-green .kpi-value { color:#059669; }
.kpi-blue  .kpi-value { color:#2563eb; }

.sec { font-size:1rem; font-weight:600; color:#111827; border-left:3px solid #10b981; padding-left:0.7rem; margin:1.8rem 0 0.8rem; }
.sec-num { color:#10b981; margin-right:0.4rem; }

.callout { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:0.9rem 1.1rem; font-size:0.85rem; color:#14532d; margin-top:0.5rem; }
.callout-warn { background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; }
.callout-blue { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a5f; }

.pred-result { background:#0a0a0a; border-radius:10px; padding:1.4rem 1.8rem; margin-top:0.8rem; }
.pred-label  { font-size:0.72rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
.pred-big    { font-size:2.8rem; font-weight:700; color:#10b981; line-height:1; margin:0.3rem 0; }
.pred-interp { font-size:0.88rem; color:#d1d5db; margin-top:0.4rem; }
.pred-caution{ font-size:0.74rem; color:#4b5563; font-style:italic; margin-top:0.6rem; }

.rec-card { background:#0a0a0a; border:1px solid #1f2937; border-radius:12px; padding:1.4rem 1.6rem; height:100%; }
.rec-card h4 { color:#10b981; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 0.5rem; }
.rec-card p  { color:#e5e7eb; font-size:0.88rem; line-height:1.6; margin:0; }
.rec-card strong { color:#ffffff; }

.quote { border-left:3px solid #10b981; padding:0.7rem 1.2rem; background:#f9fafb; border-radius:0 8px 8px 0; font-style:italic; color:#374151; font-size:0.9rem; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in ['monthly_ai_budget_usd','annual_income_usd','satisfaction_score',
               'likelihood_to_recommend','age','household_size']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    trust_num = {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
    priv_num  = {'Low':1,'Moderate':2,'High':3,'Extreme':4}
    df['trust_num']    = df['trust_level'].map(trust_num)
    df['privacy_num']  = df['privacy_concern'].map(priv_num)

    tech_occs = ['Software Developer','Engineer','Researcher','Data Analyst','Financial Analyst']
    df['is_segment'] = (
        df['occupation'].isin(tech_occs) &
        df['usage_frequency'].isin(['Daily','Multiple times daily','A few times a week','Weekly'])
    )

    trust_kw = ['fake','wrong','citation','hallucin','trust','reliable','incorrect',
                'error','made up','inaccur','mislead','mistake','verify','check','source']
    df['has_trust_concern'] = df['ai_feedback_comment'].str.lower().str.contains(
        '|'.join(trust_kw), na=False)

    inf_map = {'google':'Search engine','TikTok':'Social media','word of mouth':'Friend/family',
               'reddit':'Social media','boss':'Employer','Twitter/X':'Social media',
               'professor':'Professor/teacher'}
    df['info_source'] = df['info_source'].apply(
        lambda x: inf_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    emp_map = {'full time':'Full-time','FT':'Full-time','freelance':'Freelance',
               'contractor':'Contract','self employed':'Self-employed','part time':'Part-time'}
    df['employment_status'] = df['employment_status'].apply(
        lambda x: emp_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    return df

uploaded = st.sidebar.file_uploader("📂 Upload data_cleaned_final.csv", type=["csv"])
if uploaded:
    df = load_data(uploaded)
    st.sidebar.success(f"✅ {len(df):,} rows loaded")
else:
    try:
        df = load_data("data_cleaned_final.csv")
        st.sidebar.info("Using bundled dataset")
    except:
        st.error("Upload data_cleaned_final.csv via the sidebar.")
        st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("## Filters")
st.sidebar.caption("All charts update with these filters applied.")

age_opts   = sorted([x for x in df['age_group'].dropna().unique() if x != 'Unknown'])
tool_opts  = sorted(df['last_tool_used'].dropna().unique())
use_opts   = sorted(df['primary_ai_use_case'].dropna().unique())
stage_opts = [x for x in ['Problem recognition','Information search',
               'Evaluation of alternatives','Purchase decision','Postpurchase behavior']
              if x in df['decision_stage'].unique()]
inc_opts   = [x for x in ['Under $40K','$40K-$74K','$75K-$99K','$100K-$149K','$150K+']
              if x in df['income_group'].unique()]
sub_opts   = sorted(df['subscription_status'].dropna().unique())

sel_age   = st.sidebar.multiselect("Age Group",          age_opts,   default=age_opts)
sel_tool  = st.sidebar.multiselect("Last Tool Used",     tool_opts,  default=tool_opts)
sel_use   = st.sidebar.multiselect("AI Use Case",        use_opts,   default=use_opts)
sel_stage = st.sidebar.multiselect("Decision Stage",     stage_opts, default=stage_opts)
sel_inc   = st.sidebar.multiselect("Income Group",       inc_opts,   default=inc_opts)
sel_sub   = st.sidebar.multiselect("Subscription",       sub_opts,   default=sub_opts)

fdf = df[
    df['age_group'].isin(sel_age + ['Unknown']) &
    df['last_tool_used'].isin(sel_tool) &
    df['primary_ai_use_case'].isin(sel_use) &
    df['decision_stage'].isin(sel_stage + ['Not sure']) &
    df['income_group'].isin(sel_inc + ['Unknown']) &
    df['subscription_status'].isin(sel_sub)
]
seg = fdf[fdf['is_segment']]
rest = fdf[~fdf['is_segment']]

st.sidebar.markdown(f"**{len(fdf):,}** total &nbsp;|&nbsp; **{len(seg):,}** in segment")

# ── Colour palette ────────────────────────────────────────────────────────────
GREEN  = '#10b981'
GRAY   = '#94a3b8'
DARK   = '#1f2937'
RED    = '#ef4444'
AMBER  = '#f59e0b'
BLUE   = '#3b82f6'
TOOL_COLORS = {
    'ChatGPT/OpenAI': '#10a37f',
    'Claude':         '#cc785c',
    'Gemini':         '#4285f4',
    'Meta AI':        '#0866ff',
    'xAI Grok':       '#1da1f2',
    'Other':          '#94a3b8',
    'Unknown':        '#d1d5db',
}
SENT_COLORS = {'Positive':GREEN,'Neutral':GRAY,'Mixed':AMBER,'Negative':RED,'Ugly':'#7c3aed'}

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "01  Market Overview",
    "02  Target Segment",
    "03  Channels & Sentiment",
    "04  Predictive Models",
    "05  Recommendation",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 01 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">ECCHHO Media · Prepared for OpenAI Leadership</div>
        <div class="hero-h1">Target the people whose reputations ride on being right..</div>
        <div class="hero-sub">AI Consumer Behavior Study · N=982 cleaned respondents · Spring 2026</div>
        <div class="hero-team">Team: Elina · Hanqi · Chunfei · Hojin · Charlotte</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row — headline numbers from pitch
    paid_seg  = seg['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100
    paid_rest = rest['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100
    bud_seg   = seg['monthly_ai_budget_usd'].mean()
    bud_rest  = rest['monthly_ai_budget_usd'].mean()
    bud_lift  = (bud_seg - bud_rest) / bud_rest * 100 if bud_rest else 0
    trust_concern_pct = fdf['has_trust_concern'].mean()*100

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f'<div class="kpi kpi-green"><div class="kpi-label">Segment Paid Rate</div><div class="kpi-value">{paid_seg:.0f}%</div><div class="kpi-delta">vs {paid_rest:.0f}% rest of market</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi kpi-blue"><div class="kpi-label">Segment Avg Budget</div><div class="kpi-value">${bud_seg:.0f}</div><div class="kpi-delta">+{bud_lift:.0f}% above market</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-label">Feedback: Trust Concern</div><div class="kpi-value">{trust_concern_pct:.0f}%</div><div class="kpi-delta">of all open-ended comments</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-label">Segment Size</div><div class="kpi-value">{len(seg)}</div><div class="kpi-delta">{len(seg)/max(len(fdf),1)*100:.0f}% of surveyed market</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Chart 1: Tool market share — fragmented, contestable
    st.markdown('<div class="sec"><span class="sec-num">1.1</span>The market is fragmented — no tool has a dominant share</div>', unsafe_allow_html=True)
    c1a, c1b = st.columns([3, 2])
    with c1a:
        tool_cnt = fdf['last_tool_used'].value_counts().reset_index()
        tool_cnt.columns = ['Tool','Count']
        tool_cnt['Pct'] = (tool_cnt['Count']/tool_cnt['Count'].sum()*100).round(1)
        fig_tools = px.bar(
            tool_cnt.sort_values('Count'), x='Count', y='Tool', orientation='h',
            color='Tool', color_discrete_map=TOOL_COLORS, text='Pct',
        )
        fig_tools.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_tools.update_layout(
            height=320, showlegend=False, margin=dict(l=0,r=60,t=10,b=10),
            xaxis_title="Respondents", yaxis_title="",
            plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig_tools, use_container_width=True)

    with c1b:
        st.markdown("")
        st.markdown("")
        non_oai_seg = seg[seg['last_tool_used']!='ChatGPT/OpenAI']
        st.markdown(f"""
        <div class="callout">
        <strong>{len(non_oai_seg)/max(len(seg),1)*100:.0f}% of the target segment</strong> last used a non-OpenAI tool.<br><br>
        This is a <strong>switching opportunity</strong>, not a saturated market.
        Most respondents sit in Information Search or Evaluation of Alternatives —
        they are actively comparing, not locked in.
        </div>
        """, unsafe_allow_html=True)

        stage_dist = fdf[fdf['decision_stage'].isin(stage_opts)]['decision_stage'].value_counts(normalize=True)*100
        eval_pct = stage_dist.get('Evaluation of alternatives',0) + stage_dist.get('Information search',0)
        st.markdown(f"""
        <div class="callout callout-blue" style="margin-top:0.7rem">
        <strong>{eval_pct:.0f}%</strong> of all respondents are in Information Search or
        Evaluation of Alternatives — they are still deciding.
        </div>
        """, unsafe_allow_html=True)

    # Chart 2: Income vs occupation — the pitch's core claim
    st.markdown('<div class="sec"><span class="sec-num">1.2</span>Income barely predicts who pays — occupation does</div>', unsafe_allow_html=True)

    c2a, c2b = st.columns(2)
    with c2a:
        # Income vs paid rate scatter — show flat line
        inc_bins = pd.cut(fdf['annual_income_usd'].dropna(),
                          bins=[0,40000,75000,100000,150000,200000],
                          labels=['<$40K','$40–74K','$75–99K','$100–149K','$150K+'])
        inc_paid_df = fdf.copy()
        inc_paid_df['inc_bin'] = inc_bins
        inc_plot = inc_paid_df.groupby('inc_bin', observed=True).apply(
            lambda x: pd.Series({
                'paid_rate': x['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100,
                'n': len(x)
            })
        ).reset_index()
        fig_inc = px.bar(inc_plot, x='inc_bin', y='paid_rate',
                         color_discrete_sequence=[GRAY]*5,
                         text='paid_rate',
                         labels={'inc_bin':'Income Band','paid_rate':'% Paid Subscribers'})
        fig_inc.update_traces(texttemplate='%{text:.0f}%', textposition='outside',
                               marker_color=GRAY)
        # Flat reference line
        fig_inc.add_hline(y=inc_plot['paid_rate'].mean(), line_dash='dash',
                          line_color=RED, annotation_text='Market avg',
                          annotation_position='top right')
        fig_inc.update_layout(height=340, showlegend=False,
                               title="Paid Rate by Income Band — nearly flat",
                               plot_bgcolor='white', paper_bgcolor='white',
                               yaxis=dict(range=[0,65]), margin=dict(t=40,b=10))
        st.plotly_chart(fig_inc, use_container_width=True)

    with c2b:
        # Occupation group paid rate — highlight technical
        occ_groups = {
            'Engineer / Developer': ['Software Developer','Engineer'],
            'Researcher / Analyst': ['Researcher','Data Analyst','Financial Analyst'],
            'Business / Marketing': ['Marketing Manager','Small Business Owner','Consultant','Product Manager'],
            'Creative':             ['Content Creator','Graphic Designer','UX Designer','Journalist'],
            'Healthcare':           ['Nurse','Physician Assistant'],
            'Admin / Support':      ['Customer Support Rep','Administrative Assistant','HR Specialist'],
            'Education':            ['Teacher','K-12 Teacher'],
            'Student':              ['Student'],
        }
        rows = []
        for grp, occs in occ_groups.items():
            sub = fdf[fdf['occupation'].isin(occs)]
            if len(sub) >= 5:
                rows.append({'Group':grp, 'n':len(sub),
                    'paid_rate': sub['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100,
                    'avg_budget': sub['monthly_ai_budget_usd'].mean()})
        occ_df = pd.DataFrame(rows).sort_values('paid_rate')

        colors = [GREEN if g in ('Engineer / Developer','Researcher / Analyst') else GRAY
                  for g in occ_df['Group']]
        fig_occ = go.Figure(go.Bar(
            x=occ_df['paid_rate'], y=occ_df['Group'], orientation='h',
            marker_color=colors,
            text=occ_df['paid_rate'].apply(lambda x: f'{x:.0f}%'),
            textposition='outside'
        ))
        fig_occ.update_layout(height=340, showlegend=False,
                               title="Paid Rate by Occupation Group — technical roles lead",
                               plot_bgcolor='white', paper_bgcolor='white',
                               xaxis=dict(range=[0,65], title='% Paid Subscribers'),
                               yaxis_title='', margin=dict(t=40,r=60,b=10))
        st.plotly_chart(fig_occ, use_container_width=True)

    st.markdown("""
    <div class="callout callout-blue">
    <strong>Logistic regression finding:</strong> Technical role lifts the odds of paying by ~40%
    (odds ratio ≈ 1.41) holding income constant. Income per $10K has an odds ratio of ~0.99 — effectively zero effect.
    Occupation is the real signal.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 02 — TARGET SEGMENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:

    st.markdown('<div class="sec"><span class="sec-num">2.1</span>Segment definition — Reliability-First Technical Professionals</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    d1.markdown("""**Occupation**
- Software Developer
- Engineer
- Researcher
- Data Analyst
- Financial Analyst""")
    d2.markdown("""**Behaviour**
- Uses AI weekly or more
- Currently free, trial, or paid
- High-stakes output others see and judge""")
    d3.markdown("""**Discovery**
- Finds tools via search, news & coworkers
- Decision stage: evaluating alternatives
- Motivated by accuracy and credibility""")

    st.markdown("---")
    st.markdown('<div class="sec"><span class="sec-num">2.2</span>Segment vs market — key metrics</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Paid Rate",      f"{seg['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100:.0f}%",
                           f"Market: {fdf['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100:.0f}%"),
        ("Avg Budget",     f"${seg['monthly_ai_budget_usd'].mean():.0f}",
                           f"Market: ${fdf['monthly_ai_budget_usd'].mean():.0f}"),
        ("Avg NPS",        f"{seg['likelihood_to_recommend'].mean():.1f}",
                           f"Market: {fdf['likelihood_to_recommend'].mean():.1f}"),
        ("Avg Satisfaction",f"{seg['satisfaction_score'].mean():.2f}",
                            f"Market: {fdf['satisfaction_score'].mean():.2f}"),
        ("Avg Trust",      f"{seg['trust_num'].mean():.2f}/5",
                           f"Market: {fdf['trust_num'].mean():.2f}/5"),
    ]
    for col, (label, val, sub) in zip([m1,m2,m3,m4,m5], metrics):
        col.markdown(f'<div class="kpi kpi-green"><div class="kpi-label">{label}</div><div class="kpi-value" style="font-size:1.5rem">{val}</div><div class="kpi-delta">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c3a, c3b = st.columns(2)
    with c3a:
        # Budget by occupation group — show segment outliers
        bud_occ = fdf[fdf['occupation'].isin(
            ['Software Developer','Engineer','Researcher','Data Analyst','Financial Analyst',
             'Marketing Manager','Teacher','Nurse','Student','Content Creator','Freelancer'])
        ].copy()
        bud_occ['Segment'] = bud_occ['occupation'].isin(
            ['Software Developer','Engineer','Researcher','Data Analyst','Financial Analyst'])
        bud_occ['Group'] = bud_occ['occupation']
        fig_bud = px.box(bud_occ.dropna(subset=['monthly_ai_budget_usd']),
                         x='Group', y='monthly_ai_budget_usd',
                         color='Segment',
                         color_discrete_map={True: GREEN, False: GRAY},
                         title="Monthly AI Budget by Occupation — segment (green) pays more",
                         labels={'monthly_ai_budget_usd':'Monthly Budget ($)','Group':''})
        fig_bud.update_layout(height=400, showlegend=True,
                               xaxis_tickangle=-30, plot_bgcolor='white',
                               paper_bgcolor='white', margin=dict(t=40,b=80))
        st.plotly_chart(fig_bud, use_container_width=True)

    with c3b:
        # Decision stage distribution: segment vs market side by side
        stage_order = ['Problem recognition','Information search',
                       'Evaluation of alternatives','Purchase decision','Postpurchase behavior']
        seg_stages  = seg[seg['decision_stage'].isin(stage_order)]['decision_stage'].value_counts(normalize=True)*100
        mkt_stages  = fdf[fdf['decision_stage'].isin(stage_order)]['decision_stage'].value_counts(normalize=True)*100

        stage_df = pd.DataFrame({
            'Stage': stage_order,
            'Segment': [seg_stages.get(s,0) for s in stage_order],
            'Full Market': [mkt_stages.get(s,0) for s in stage_order],
        })
        fig_stage = go.Figure()
        fig_stage.add_trace(go.Bar(name='Segment', x=stage_df['Stage'], y=stage_df['Segment'],
                                    marker_color=GREEN, text=stage_df['Segment'].apply(lambda x: f'{x:.0f}%'),
                                    textposition='outside'))
        fig_stage.add_trace(go.Bar(name='Full Market', x=stage_df['Stage'], y=stage_df['Full Market'],
                                    marker_color=GRAY, text=stage_df['Full Market'].apply(lambda x: f'{x:.0f}%'),
                                    textposition='outside'))
        fig_stage.update_layout(barmode='group', height=400,
                                 title="Decision Stage: Segment vs Market — both skew toward evaluation",
                                 plot_bgcolor='white', paper_bgcolor='white',
                                 xaxis_tickangle=-20, yaxis_title='% of Group',
                                 legend=dict(orientation='h', x=0.3, y=1.1),
                                 margin=dict(t=50,b=80))
        st.plotly_chart(fig_stage, use_container_width=True)

    st.markdown("""
    <div class="callout">
    The segment skews toward <strong>Evaluation of Alternatives and Information Search</strong> —
    they are actively comparing tools and have not committed. This is the optimal window for a
    targeted paid-seat conversion campaign.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 03 — CHANNELS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:

    st.markdown('<div class="sec"><span class="sec-num">3.1</span>Where the segment discovers tools — channel validation</div>', unsafe_allow_html=True)

    c4a, c4b = st.columns(2)
    with c4a:
        # Info source: segment vs market
        sources = ['Search engine','News article','Coworker','YouTube','Employer',
                   'Podcast','Friend/family','Social media','Ad']
        seg_src = seg['info_source'].value_counts(normalize=True).reindex(sources, fill_value=0)*100
        mkt_src = fdf['info_source'].value_counts(normalize=True).reindex(sources, fill_value=0)*100

        src_df = pd.DataFrame({
            'Channel': sources,
            'Segment': seg_src.values,
            'Full Market': mkt_src.values,
        }).sort_values('Segment', ascending=True)

        fig_src = go.Figure()
        fig_src.add_trace(go.Bar(name='Segment', y=src_df['Channel'], x=src_df['Segment'],
                                  orientation='h', marker_color=GREEN))
        fig_src.add_trace(go.Bar(name='Full Market', y=src_df['Channel'], x=src_df['Full Market'],
                                  orientation='h', marker_color=GRAY))
        fig_src.update_layout(barmode='group', height=400,
                               title="Info Source: Segment over-indexes Search, Coworker, News",
                               xaxis_title='% of Group', yaxis_title='',
                               plot_bgcolor='white', paper_bgcolor='white',
                               legend=dict(orientation='h', x=0.3, y=1.12),
                               margin=dict(t=50,r=20))
        st.plotly_chart(fig_src, use_container_width=True)
        st.markdown("""
        <div class="callout">
        The segment over-indexes <strong>Search Engine, Coworker, and News Article</strong> versus the broader market.
        This validates high-intent search ads + technical newsletters + a workplace referral motion as the right channel mix.
        </div>
        """, unsafe_allow_html=True)

    with c4b:
        # Sentiment x tool — show that trust concern is universal, not tool-specific
        sent_tool = fdf.groupby(['last_tool_used','feedback_sentiment']).size().reset_index(name='Count')
        sent_total = sent_tool.groupby('last_tool_used')['Count'].transform('sum')
        sent_tool['Pct'] = sent_tool['Count']/sent_total*100

        fig_sent = px.bar(sent_tool, x='Pct', y='last_tool_used', color='feedback_sentiment',
                          barmode='stack', orientation='h',
                          color_discrete_map=SENT_COLORS,
                          title="Sentiment by Tool — negative/ugly share is market-wide",
                          labels={'Pct':'% share','last_tool_used':'Tool'})
        fig_sent.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                               legend_title='Sentiment', xaxis_title='% of Respondents',
                               yaxis_title='', margin=dict(t=40,r=20))
        st.plotly_chart(fig_sent, use_container_width=True)
        st.markdown("""
        <div class="callout callout-warn">
        Negative and Ugly sentiment exists across <em>every</em> tool — this is a
        market-wide reliability problem, not a single-brand issue.
        The first brand to credibly solve it wins the professional segment.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec"><span class="sec-num">3.2</span>The trust-reliability gap — open feedback analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="quote">"It made me look bad in a meeting because I trusted it."<br>
    <span style="font-size:0.8rem;color:#6b7280">— Verbatim survey feedback, reputational fear in the customer's own words</span>
    </div>
    """, unsafe_allow_html=True)

    c5a, c5b = st.columns([2, 3])
    with c5a:
        trust_concern_by_tool = fdf.groupby('last_tool_used')['has_trust_concern'].mean()*100
        tc_df = trust_concern_by_tool.reset_index()
        tc_df.columns = ['Tool','% Trust Concern']
        fig_tc = px.bar(tc_df.sort_values('% Trust Concern'), x='% Trust Concern', y='Tool',
                        orientation='h', color='% Trust Concern',
                        color_continuous_scale=['#f0fdf4','#10b981','#065f46'],
                        title="% Feedback Mentioning Trust/Reliability — by Tool")
        fig_tc.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                              coloraxis_showscale=False, margin=dict(t=40,r=20))
        st.plotly_chart(fig_tc, use_container_width=True)

    with c5b:
        # Trust score vs satisfaction — coloured by segment flag
        sat_trust_df = fdf.dropna(subset=['trust_num','satisfaction_score']).copy()
        sat_trust_df['Who'] = sat_trust_df['is_segment'].map(
            {True:'Target Segment', False:'Rest of Market'})
        fig_ts = px.scatter(sat_trust_df, x='trust_num', y='satisfaction_score',
                            color='Who', opacity=0.45, trendline='ols',
                            color_discrete_map={'Target Segment':GREEN,'Rest of Market':GRAY},
                            title="Trust Score vs Satisfaction — segment benefits most from trust lift",
                            labels={'trust_num':'Trust Level (1=Very Low, 5=Very High)',
                                    'satisfaction_score':'Satisfaction Score (1-10)'})
        fig_ts.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(t=40,r=20))
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("""
    <div class="callout callout-blue">
    <strong>Business implication:</strong> Whoever resolves perceived risk — functional (fake citations, wrong facts)
    and reputational (looking bad in front of colleagues) — earns the subscription.
    The technical segment is most sensitive to this gap because the stakes of an error are highest
    when their work is judged by others.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 04 — PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:

    st.markdown('<div class="sec"><span class="sec-num">4.1</span>Model 1 — Predict likelihood to pay (Logistic Regression)</div>', unsafe_allow_html=True)
    st.markdown("Isolates the effect of occupation type vs income on paid subscription probability — the core claim of the pitch.")

    @st.cache_data
    def train_logreg(df):
        tech_occs = ['Software Developer','Engineer','Researcher','Data Analyst','Financial Analyst']
        mdf = df.dropna(subset=['annual_income_usd','satisfaction_score']).copy()
        mdf['paid']      = mdf['subscription_status'].isin(['Paid individual','Paid work/school']).astype(int)
        mdf['tech_role'] = mdf['occupation'].isin(tech_occs).astype(int)
        freq_enc = {'Tried once':1,'Rarely':2,'Monthly':3,'Weekly':4,
                    'A few times a week':5,'Daily':6,'Multiple times daily':7}
        mdf['freq_num']  = mdf['usage_frequency'].map(freq_enc).fillna(4)
        trust_enc = {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
        mdf['trust_num2']= mdf['trust_level'].map(trust_enc).fillna(3)
        edu_enc = {'High school':1,'Some college':2,'Trade/Certificate':2,'Associate degree':3,
                   "Bachelor's degree":4,"Master's degree":5,'Doctorate':6}
        mdf['edu_num']   = mdf['prior_education'].map(edu_enc).fillna(3)
        feats = ['annual_income_usd','tech_role','freq_num','trust_num2','edu_num','satisfaction_score']
        X = mdf[feats]; y = mdf['paid']
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, y)
        import numpy as np
        odds = dict(zip(feats, np.exp(lr.coef_[0])))
        return lr, feats, odds, freq_enc, trust_enc, edu_enc

    lr_model, feats_lr, odds_dict, freq_enc, trust_enc, edu_enc = train_logreg(df)

    # Odds ratio chart
    odds_plot = pd.DataFrame({
        'Feature': ['Technical Role','Usage Frequency','Trust Level','Education','Satisfaction','Income ($10K)'],
        'Odds Ratio': [
            odds_dict['tech_role'],
            odds_dict['freq_num'],
            odds_dict['trust_num2'],
            odds_dict['edu_num'],
            odds_dict['satisfaction_score'],
            np.exp(odds_dict['annual_income_usd']*10000),
        ]
    }).sort_values('Odds Ratio')
    odds_plot['Color'] = odds_plot['Odds Ratio'].apply(lambda x: GREEN if x > 1.1 else RED if x < 0.9 else GRAY)
    odds_plot['Label'] = odds_plot['Odds Ratio'].apply(lambda x: f'{x:.2f}x')

    fig_odds = go.Figure(go.Bar(
        x=odds_plot['Odds Ratio'], y=odds_plot['Feature'], orientation='h',
        marker_color=odds_plot['Color'], text=odds_plot['Label'], textposition='outside'
    ))
    fig_odds.add_vline(x=1, line_dash='dash', line_color='black', line_width=1)
    fig_odds.update_layout(height=300, title="Odds Ratios — what actually predicts paying for AI",
                           plot_bgcolor='white', paper_bgcolor='white',
                           xaxis=dict(title='Odds Ratio (>1 = increases likelihood to pay)',range=[0,2.2]),
                           yaxis_title='', margin=dict(t=40,r=80))
    st.plotly_chart(fig_odds, use_container_width=True)

    st.markdown("""
    <div class="callout callout-blue">
    Technical role has the largest positive odds ratio. Income per $10K has an odds ratio near 1.0 — 
    confirming the pitch headline: <strong>income barely predicts who pays; occupation does.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec"><span class="sec-num">4.2</span>Interactive predictor — estimate upgrade probability for any consumer profile</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        l_income  = st.slider("Annual Income ($)", 20000, 160000, 90000, step=5000)
        l_tech    = st.selectbox("Occupation Type", ["Technical Role (Engineer/Developer/Analyst/Researcher)",
                                                      "Non-technical Role"])
    with p2:
        l_freq    = st.selectbox("Usage Frequency", list(freq_enc.keys()), index=5)
        l_trust   = st.selectbox("Trust Level", list(trust_enc.keys()), index=3)
    with p3:
        l_edu     = st.selectbox("Education", list(edu_enc.keys()), index=3)
        l_sat     = st.slider("Satisfaction Score", 1, 10, 7)

    l_tech_bin = 1 if "Technical" in l_tech else 0
    Xl = np.array([[l_income, l_tech_bin, freq_enc.get(l_freq,4),
                    trust_enc.get(l_trust,3), edu_enc.get(l_edu,3), l_sat]])
    pay_prob = lr_model.predict_proba(Xl)[0][1]

    col_gauge, col_interp = st.columns([1,1])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pay_prob*100, 1),
            number={'suffix':'%','font':{'size':42,'color':GREEN}},
            title={'text':'Upgrade Probability','font':{'size':14,'color':'#6b7280'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#e5e7eb'},
                'bar':{'color': GREEN if pay_prob>=0.5 else AMBER if pay_prob>=0.3 else RED},
                'bgcolor':'white',
                'steps':[
                    {'range':[0,30],'color':'#fef2f2'},
                    {'range':[30,60],'color':'#fffbeb'},
                    {'range':[60,100],'color':'#f0fdf4'},
                ],
                'threshold':{'line':{'color':'#111827','width':3},'thickness':0.8,'value':50}
            }
        ))
        fig_gauge.update_layout(height=280, paper_bgcolor='white', margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_interp:
        if pay_prob >= 0.60:
            msg = "🟢 <strong>High-priority upgrade target.</strong> Recommend a personalised ChatGPT Plus offer or team-seat trial. This profile matches the core segment — deploy direct outreach through employer partnerships or technical community ads."
        elif pay_prob >= 0.40:
            msg = "🟡 <strong>Warm lead.</strong> Nurture with verification-focused content — benchmarks, source demos. A 30-day trial CTA with a low-friction onboarding flow should convert this profile."
        else:
            msg = "🔴 <strong>Low conversion likelihood right now.</strong> Focus on building trust and demonstrating accuracy before any upsell attempt. Educational content through their preferred channel first."

        st.markdown(f"""
        <div class="pred-result">
            <div class="pred-label">Predicted upgrade probability</div>
            <div class="pred-big">{pay_prob*100:.1f}%</div>
            <div class="pred-interp">{msg}</div>
            <div class="pred-caution">⚠️ Based on synthetic survey data. Supports, does not replace, human judgment.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Model 2 — satisfaction predictor (simpler, visual)
    st.markdown('<div class="sec"><span class="sec-num">4.3</span>Model 2 — Predict satisfaction score (Random Forest)</div>', unsafe_allow_html=True)
    st.markdown("Shows that trust level is the strongest lever for satisfaction — validating the verification-first message.")

    @st.cache_data
    def train_rf_sat(df):
        freq_enc2 = {'Tried once':1,'Rarely':2,'Monthly':3,'Weekly':4,
                     'A few times a week':5,'Daily':6,'Multiple times daily':7}
        trust_enc2= {'Very low':1,'Low':2,'Moderate':3,'High':4,'Very high':5}
        mdf = df.dropna(subset=['satisfaction_score','age','annual_income_usd','monthly_ai_budget_usd']).copy()
        mdf['freq_num']  = mdf['usage_frequency'].map(freq_enc2).fillna(4)
        mdf['trust_num2']= mdf['trust_level'].map(trust_enc2).fillna(3)
        le = LabelEncoder()
        mdf['use_enc']   = le.fit_transform(mdf['primary_ai_use_case'].fillna('Other'))
        feats = ['age','annual_income_usd','monthly_ai_budget_usd','freq_num','trust_num2','use_enc']
        X = mdf[feats]; y = mdf['satisfaction_score']
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        r2 = r2_score(y, rf.predict(X))
        return rf, le, feats, freq_enc2, trust_enc2, r2

    rf_model, le_rf, feats_rf, freq_enc2, trust_enc2, r2 = train_rf_sat(df)

    imp_df = pd.DataFrame({'Feature':feats_rf,'Importance':rf_model.feature_importances_})
    imp_df['Feature'] = imp_df['Feature'].map({'age':'Age','annual_income_usd':'Income',
        'monthly_ai_budget_usd':'Monthly Budget','freq_num':'Usage Frequency',
        'trust_num2':'Trust Level','use_enc':'Use Case'})
    imp_df = imp_df.sort_values('Importance')
    imp_df['Color'] = imp_df['Feature'].apply(lambda x: GREEN if x == 'Trust Level' else BLUE if x == 'Usage Frequency' else GRAY)

    fig_imp = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker_color=imp_df['Color'],
        text=imp_df['Importance'].apply(lambda x: f'{x:.2f}'),
        textposition='outside'
    ))
    fig_imp.update_layout(height=300, title=f"Feature Importance for Satisfaction (R²={r2:.2f}) — trust is the #1 driver",
                          plot_bgcolor='white', paper_bgcolor='white',
                          xaxis_title='Importance Score', yaxis_title='',
                          margin=dict(t=40,r=60))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("""
    <div class="callout">
    Trust level is the strongest predictor of satisfaction score in the random forest model.
    This directly supports the "Ask. Check. Ship." message — improving perceived reliability
    is the fastest lever to raise satisfaction and reduce churn.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 05 — RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Final Recommendation · ECCHHO Media for OpenAI</div>
        <div class="hero-h1">"Ask. Check. Ship."</div>
        <div class="hero-sub">Make verifiability the message, technical professionals the audience, and the team seat the conversion.</div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    for col, (h, body) in zip([r1,r2,r3,r4], [
        ("🎯 Segment",
         "<strong>Reliability-First Technical Professionals</strong> — engineers, developers, researchers, analysts. Weekly+ users with high-stakes, peer-visible output."),
        ("📢 Channel",
         "<strong>High-intent search ads</strong> + technical forums and newsletters + <strong>workplace referral motion</strong> (coworker channel over-indexes in segment)."),
        ("💬 Message",
         "<strong>Promise verifiability, not infallibility.</strong> 'Ask. Check. Ship.' — checkable, cited answers via ChatGPT Search. Never say 'trust it.'"),
        ("⚡ Action",
         "<strong>Convert free & competitor users into paid team seats.</strong> CTA: 'Start a team trial.' Anchor to a specific workflow — e.g. project brief, code review."),
    ]):
        col.markdown(f'<div class="rec-card"><h4>{h}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec"><span class="sec-num">5.1</span>Evidence summary — four data points from the dashboard</div>', unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    evidence = [
        ("Paid Rate", f"Technical roles convert at {seg['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100:.0f}% vs {rest['subscription_status'].isin(['Paid individual','Paid work/school']).mean()*100:.0f}% for the rest — a meaningful lift that income alone cannot explain (odds ratio ≈ 1.41 vs ≈ 1.00)."),
        ("Budget", f"The segment spends ${seg['monthly_ai_budget_usd'].mean():.0f}/month on average vs ${rest['monthly_ai_budget_usd'].mean():.0f} for the market — {(seg['monthly_ai_budget_usd'].mean()/rest['monthly_ai_budget_usd'].mean()-1)*100:.0f}% higher willingness to pay."),
        ("Trust Concern", f"{fdf['has_trust_concern'].mean()*100:.0f}% of all open-ended feedback mentions a trust or reliability concern — and the satisfaction scatter confirms trust is the #1 predictor of satisfaction score in the random forest model."),
        ("Switching Opportunity", f"{len(seg[seg['last_tool_used']!='ChatGPT/OpenAI'])/max(len(seg),1)*100:.0f}% of the target segment last used a non-OpenAI tool, and {(seg['decision_stage'].isin(['Evaluation of alternatives','Information search'])).mean()*100:.0f}% are in active evaluation or search stages — they are still deciding."),
    ]
    for col, (title, text) in zip([e1,e2,e3,e4], evidence):
        col.markdown(f"""<div class="kpi" style="text-align:left;height:100%">
            <div class="kpi-label" style="color:#10b981">{title}</div>
            <div style="font-size:0.85rem;color:#374151;margin-top:0.5rem;line-height:1.5">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-blue">
    <strong>A note on rigor:</strong> The promise is scoped to ChatGPT Search's sourced answers —
    steer users into sourced mode and never imply every reply is cited.
    Reliability is a market-wide concern, but the stakes of an error are highest for professionals
    whose work is judged by others, which is why the message lands hardest here.<br><br>
    <em>Methodology: 982-respondent cleaned AI consumer survey. Segment lift from group comparisons;
    driver isolated via logistic regression (role vs income, income held constant).
    Feedback themes from keyword-coded open-ended comments.
    Figures from synthetic survey data — intended to support, not replace, human judgment.</em>
    </div>
    """, unsafe_allow_html=True)
