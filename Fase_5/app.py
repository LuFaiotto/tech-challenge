# Imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import shap

import google.generativeai as genai

st.markdown("""
    <style>
    p {text-align: justify;}

    section[data-testid="stSidebar"] {
        width: 440px !important;
    }
    </style>
    """, unsafe_allow_html=True)

numeric_cols_to_clean = [
    "IAA", "IDA", "IEG", "IPS", "IPV", "IPP", "IAN",
    "INDE 2022", "INDE 2023", "INDE 2024"
]

def clean_numeric_column(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, 'PEDE_Completo_Normalizado.csv')
df = pd.read_csv(path, sep=';', engine='python', encoding='latin1')
pd.options.display.float_format = '{:,.2f}'.format

for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])
        
# Conexão Gemini API
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except userdata.SecretNotFoundError:
    GEMINI_API_KEY = None # Se não encontrar, define como None
    st.warning("ATENÇÃO: A chave 'GEMINI_API_KEY' não foi encontrada nos Secrets do Colab.")
    st.warning("Por favor, adicione sua chave da API do Gemini como um Secret do Colab para usar os insights do Gemini.")
    st.warning("Vá em 'Secrets' (ícone de chave no painel esquerdo) e adicione 'GEMINI_API_KEY' com sua chave.")
except Exception as e:
    GEMINI_API_KEY = None
    st.error(f"Erro ao configurar a API do Gemini: {e}")

def generate_gemini_insights(prompt):
    if GEMINI_API_KEY is None:
        return "Erro: A chave da API do Gemini não está configurada. Por favor, configure-a nos Secrets do Colab."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        # Verifica se há conteúdo na resposta antes de tentar acessá-lo
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return f"A API do Gemini não retornou conteúdo. Possíveis bloqueios: {response.prompt_feedback}"
    except Exception as e:
        return f"Erro ao gerar insights com Gemini: {e}"

def plot_reg_with_45_deg_line(data, x_col, y_col, title_pt, x_label_pt, y_label_pt):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_col, y=y_col, data=data, scatter_kws={"s": 50, "alpha": 0.7},
                line_kws={"color": "red", "label": "Linha de Regressão"})
    plt.title(title_pt)
    plt.xlabel(x_label_pt)
    plt.ylabel(y_label_pt)
    
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, color='purple', linestyle='--', linewidth=1, label='Linha de 45°')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def analyze_and_plot_queda_streamlit(data, year_pair_str, ips_col='IPS', other_cols=['Idade', 'Sexo', 'IEG', 'IDA']):
    queda_col = f'queda_{year_pair_str}'
    delta_inde_col = f'delta_INDE_{year_pair_str}'

    st.subheader(f"Análise para a Queda {year_pair_str}:")

    results_for_prompt = {}

    st.markdown(f"**IPS médio (queda {year_pair_str} vs. sem queda):**")
    ips_mean_stats = data.groupby(queda_col)[ips_col].agg(['count', 'mean', 'std'])
    st.dataframe(ips_mean_stats)
    results_for_prompt['ips_mean_stats'] = ips_mean_stats.to_string()
    st.markdown("""
    Esta tabela compara a média, desvio padrão e contagem de alunos do IPS entre os grupos que sofreram queda de desempenho e os que não. Uma diferença significativa nas médias pode indicar que o IPS está relacionado com a ocorrência de quedas.
    """)

    grp1 = data.loc[data[queda_col] == True, ips_col].dropna()
    grp0 = data.loc[data[queda_col] == False, ips_col].dropna()
    if len(grp1) > 1 and len(grp0) > 1:
        tstat, pval = ttest_ind(grp1, grp0, equal_var=False)
        st.write(f"**Teste T de IPS (queda vs. sem queda):** t = {tstat:.3f}, p = {pval:.4f}")
        results_for_prompt['ttest_results'] = f"t = {tstat:.3f}, p = {pval:.4f}"
        st.markdown(f"""
        O Teste T avalia se a diferença nas médias de IPS entre os grupos com e sem queda é estatisticamente significativa. Um valor-p baixo (geralmente < 0.05) sugere que a diferença observada é improvável de ter ocorrido por acaso.
        """)
    else:
        st.write(f"Não há dados suficientes para realizar o Teste T para {year_pair_str}.")
        results_for_prompt['ttest_results'] = f"Não há dados suficientes para realizar o Teste T para {year_pair_str}."

    # Regressão logística
    model_df = data[[queda_col, ips_col] + other_cols].copy()
    model_df['Sexo_M'] = model_df['Sexo'].str.lower().str.contains('masc').astype(int)
    model_df = model_df.dropna()
    if not model_df.empty:
        y = model_df[queda_col].astype(int)
        X = model_df[[ips_col, 'Idade', 'Sexo_M', 'IEG', 'IDA']].astype(float)
        X = sm.add_constant(X)
        try:
            logit = sm.Logit(y, X).fit(disp=0)
            st.markdown(f"**Regressão Logística ({queda_col} ~ {ips_col} + covariáveis):**")
            
            summary_df = pd.DataFrame({
                'Coeficiente': logit.params,
                'Erro Padrão': logit.bse,
                'Z-valor': logit.tvalues,
                'P>|z|': logit.pvalues
            })
            st.dataframe(summary_df.applymap(lambda x: f'{x:.3f}'))
            
            results_for_prompt['logit_summary'] = summary_df.applymap(lambda x: f'{x:.3f}').to_string()
            st.markdown("""
            A Regressão Logística modela a probabilidade de ocorrência de queda. Os coeficientes indicam a direção e a força da relação de cada variável com a chance de queda. Coeficientes negativos para o IPS, por exemplo, sugerem que um IPS mais alto está associado a uma menor probabilidade de queda, controlando por outras variáveis. Os p-valores indicam a significância estatística de cada coeficiente.
            """)
        except Exception as e:
            st.write(f"Erro ao executar a Regressão Logística para {year_pair_str}: {e}")
            results_for_prompt['logit_summary'] = f"Erro ao executar a Regressão Logística: {e}"
    else:
        st.write(f"Não há dados suficientes para a Regressão Logística para {year_pair_str}.")
        results_for_prompt['logit_summary'] = f"Não há dados suficientes para a Regressão Logística para {year_pair_str}."

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=ips_col, y=delta_inde_col, hue=queda_col, palette='Set1', alpha=0.7, ax=ax1)
    sns.regplot(data=data, x=ips_col, y=delta_inde_col, scatter=False, color='gray', lowess=True, line_kws={'linestyle': '--'}, ax=ax1)
    ax1.axhline(0, color='k', linestyle='--', linewidth=1)
    ax1.set_title(f'IPS vs Variação do INDE ({year_pair_str})')
    ax1.set_xlabel('IPS (Aspectos Psicossociais)')
    ax1.set_ylabel(f'Variação INDE ({year_pair_str})')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title=f'Queda {year_pair_str}', loc='best')
    st.pyplot(fig1)
    plt.close(fig1)
    st.markdown("""
    O gráfico de dispersão mostra a relação entre o IPS e a mudança no INDE. Pontos abaixo da linha horizontal de zero indicam uma queda no INDE. A linha de regressão (cinza tracejada) mostra a tendência geral. Se o IPS estiver negativamente correlacionado com a queda, esperamos ver mais pontos vermelhos (queda) com valores mais baixos de IPS, e vice-versa.
    """)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x=queda_col, y=ips_col, hue=queda_col, palette='pastel', legend=False, ax=ax2)
    ax2.set_title(f'Distribuição de IPS por Ocorrência de Queda ({year_pair_str})')
    ax2.set_xlabel(f'Queda {year_pair_str} (Verdadeiro/Falso)')
    ax2.set_ylabel('IPS (Aspectos Psicossociais)')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown("""
    Este gráfico compara a distribuição do IPS para alunos que tiveram uma queda de desempenho (Verdadeiro) e aqueles que não tiveram (Falso). As caixas mostram a mediana, quartis e possíveis outliers, permitindo visualizar a diferença na distribuição do IPS entre os dois grupos.
    """)

    return results_for_prompt

def display_question_1(df):
    st.header("Pergunta 1 - Adequação do nível (IAN)")
    st.markdown("***Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?***")

    st.subheader("Distribuição do INDE 2024")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["INDE 2024"].dropna(), bins=20)
        ax.set_title("Distribuição do INDE")
        ax.set_xlabel("INDE")
        ax.set_ylabel("Quantidade")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.metric("Média INDE", f"{df['INDE 2024'].mean():.2f}")
        st.metric("Desvio Padrão", f"{df['INDE 2024'].std():.2f}")

    st.write("Observa-se que a média do INDE é 7,4, com desvio padrão de 1,01, indicando que os dados estão relativamente concentrados próximos à média. Isso demonstra que o desempenho geral dos alunos é estável, sem variações extremas muito acentuadas. A concentração dos valores em torno da faixa 7–8 indica que a maioria dos alunos apresenta desempenho considerado bom, com pequena dispersão entre os resultados.")

def display_question_2(df):
    st.header("Pergunta 2: Desempenho acadêmico (IDA)")
    st.markdown("***O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?***")

    # --- Evolução Média do INDE ---
    st.subheader("Evolução Média do INDE")
    inde_melt = df.melt(
        id_vars=["RA", "Fase"],
        value_vars=["INDE 2022", "INDE 2023", "INDE 2024"],
        var_name="Ano",
        value_name="INDE"
    )

    inde_melt["Ano"] = inde_melt["Ano"].str.replace("INDE ", "").astype(int)

    media_inde = inde_melt.groupby("Ano")["INDE"].mean().reset_index()

    fig_evol, ax_evol = plt.subplots(figsize=(8, 4))
    ax_evol.plot(media_inde["Ano"], media_inde["INDE"], marker="o")
    ax_evol.set_title("Evolução Média do INDE 2022–2024")
    ax_evol.set_xlabel("Ano")
    ax_evol.set_ylabel("Média INDE")
    ax_evol.set_xticks(media_inde["Ano"])
    ax_evol.grid(alpha=0.3)
    st.pyplot(fig_evol)
    plt.close(fig_evol)

    st.write("Observa-se tendência de crescimento no desempenho médio ao longo do tempo, indicando melhora progressiva dos resultados acadêmicos. Esse comportamento sugere que as estratégias educacionais aplicadas podem estar contribuindo positivamente para o avanço dos indicadores. A evolução consistente reforça um cenário de melhoria gradual no desempenho institucional.")

    # --- Cálculo do Índice de Volatilidade de Aprendizagem (IVA) ---
    st.subheader("Índice de Volatilidade de Aprendizagem (IVA) por Fase")
    st.markdown("O IVA é o desvio padrão do INDE de cada aluno ao longo dos anos, agrupado por Fase.")

    inde_pivot = inde_melt.pivot_table(index=['RA', 'Fase'], columns='Ano', values='INDE')

    inde_pivot['IVA_Aluno'] = inde_pivot.std(axis=1)

    iva_por_aluno = inde_pivot.reset_index()[['RA', 'Fase', 'IVA_Aluno']].dropna(subset=['IVA_Aluno'])

    iva_por_fase = iva_por_aluno.groupby('Fase')['IVA_Aluno'].mean().reset_index()
    iva_por_fase = iva_por_fase.sort_values(by='IVA_Aluno', ascending=False)

    st.write("**IVA Médio por Fase:**")
    st.dataframe(iva_por_fase)

    if not iva_por_fase.empty:
        fase_maior_oscilacao = iva_por_fase.iloc[0]
        st.write(f"**A Fase com a maior oscilação no Índice de Volatilidade de Aprendizagem (IVA) é a Fase {fase_maior_oscilacao['Fase']} com um IVA médio de {fase_maior_oscilacao['IVA_Aluno']:.2f}.**")
    else:
        st.write("Não foi possível calcular o IVA por Fase devido a dados insuficientes.")
        fase_maior_oscilacao = None

    # --- Geração de Insights com a API do Gemini ---
    # st.subheader("Insights do Gemini sobre a Evolução do INDE e Volatilidade")

    prompt_gemini = f"""
    Analise a evolução média do Índice de Desenvolvimento do Aluno (INDE) ao longo de 2022, 2023 e 2024, e o Índice de Volatilidade de Aprendizagem (IVA) por Fase, com base nos seguintes dados:

    **1. Evolução Média do INDE:**
    ```
    {media_inde.to_string()}
    ```

    **2. IVA Médio por Fase:**
    ```
    {iva_por_fase.to_string()}
    ```

    """
    if fase_maior_oscilacao is not None:
        prompt_gemini += f"A Fase com a maior oscilação no Índice de Volatilidade de Aprendizagem (IVA) é a Fase {fase_maior_oscilacao['Fase']} com um IVA médio de {fase_maior_oscilacao['IVA_Aluno']:.2f}.\n\n"

    prompt_gemini += f"""
    Com base nesses dados, por favor, forneça:
    1. Uma análise das tendências da evolução do INDE e suas implicações para o programa 'Passos Mágicos'.
    2. Identifique quais fases apresentam maior volatilidade e discuta as possíveis causas para essa instabilidade no desempenho dos alunos.
    3. Sugira ações estratégicas e intervenções práticas que a instituição pode implementar para estabilizar o desempenho dos alunos nas fases mais voláteis e para promover uma evolução consistente do INDE.
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

def display_question_3(df):
    st.header("Engajamento nas atividades (IEG)")
    st.markdown("***O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores de desempenho (IDA) e do ponto de virada (IPV)?***")

    # Calcula a média do INDE para os anos 2022, 2023 e 2024
    df["INDE_Media"] = df[["INDE 2022", "INDE 2023", "INDE 2024"]].mean(axis=1)

    # Agrupa por Fase e calcula a média do INDE_Media
    media_fase = (
        df.groupby("Fase")["INDE_Media"]
        .mean()
        .reset_index()
    )

    # Converte 'Fase' para numérico e remove NaNs
    media_fase["Fase"] = pd.to_numeric(media_fase["Fase"], errors="coerce")
    media_fase = media_fase.dropna(subset=["Fase"])
    media_fase = media_fase.sort_values("Fase")

    # Cria o gráfico de barras
    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.bar(
        media_fase["Fase"],
        media_fase["INDE_Media"],
        width=0.6
    )

    ax3.set_xticks(media_fase["Fase"])
    ax3.set_xticklabels(media_fase["Fase"].astype(int))
    ax3.set_title("Média Geral do INDE por Fase", fontsize=14, weight="bold")
    ax3.set_xlabel("Fase")
    ax3.set_ylabel("Média INDE")
    ax3.grid(axis="y", alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

    st.write("Observa-se que as fases possuem desempenho relativamente equilibrado, com pequenas variações entre elas. Não há discrepâncias extremas, o que indica homogeneidade no desempenho entre os grupos.")
    st.write("Algumas fases apresentam médias ligeiramente superiores, podendo indicar melhor adaptação ao conteúdo ou maturidade acadêmica maior. De forma geral, o padrão é estável entre as fases.")
    
    df_q3 = df.dropna(subset=["IEG", "IPV"]).copy()

    bins = np.arange(0, 11, 1)
    labels = [f'{i}-{i+1}' for i in range(10)]
    df_q3['IEG_Faixa'] = pd.cut(df_q3['IEG'], bins=bins, labels=labels, right=False, include_lowest=True)

    if 'N_Atingiu PV' in df_q3.columns:
        df_q3['Atingiu_PV'] = df_q3['N_Atingiu PV']
    else:
        st.warning("Coluna 'N_Atingiu PV' não encontrada. Usando IPV > 0.5 como proxy para 'Atingiu_PV'.")
        df_q3['Atingiu_PV'] = (df_q3['IPV'] > df_q3['IPV'].median()).astype(int)

    pv_por_ieg_faixa = df_q3.groupby('IEG_Faixa')['Atingiu_PV'].mean().reset_index()
    pv_por_ieg_faixa['Percentual_PV'] = pv_por_ieg_faixa['Atingiu_PV'] * 100

    st.subheader("Percentual de Alunos que Atingiram o Ponto de Virada por Faixa de IEG")
    st.dataframe(pv_por_ieg_faixa)

    # Visualizar a relação
    fig_ieg_pv, ax_ieg_pv = plt.subplots(figsize=(10, 6))
    sns.barplot(x='IEG_Faixa', y='Percentual_PV', data=pv_por_ieg_faixa, palette='viridis', ax=ax_ieg_pv)
    ax_ieg_pv.set_title('Percentual de Ponto de Virada por Faixa de Engajamento (IEG)')
    ax_ieg_pv.set_xlabel('Faixa de IEG')
    ax_ieg_pv.set_ylabel('Percentual de Alunos com Ponto de Virada (%)')
    ax_ieg_pv.tick_params(axis='x', rotation=45)
    ax_ieg_pv.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig_ieg_pv)
    plt.close(fig_ieg_pv)

    # Usando a mediana do IEG de todos os alunos como um limiar mais significativo.
    ieg_cut_off_value = df_q3['IEG'].median()
    
    st.subheader("Limiar de Engajamento (IEG) Sugerido")
    st.markdown(f"Com base na análise, o **IEG mediano** entre todos os alunos é **{ieg_cut_off_value:.2f}**. Alunos com IEG abaixo deste valor podem ser considerados com engajamento mais baixo, o que pode impactar sua probabilidade de atingir o 'Ponto de Virada'.")

    if ieg_cut_off_value is not None:
        alunos_abaixo_limiar = df_q3[df_q3['IEG'] < ieg_cut_off_value]
        st.warning(f"**Alerta**: Há {len(alunos_abaixo_limiar)} alunos com IEG abaixo do limiar sugerido de {ieg_cut_off_value:.2f}. Estes alunos podem precisar de intervenções específicas para aumentar o engajamento.")
    else:
        st.info("Não foi possível determinar um limiar numérico claro de IEG para o alerta.")

    # --- Geração de Insights com a API do Gemini ---
    # st.subheader("Insights do Gemini sobre o Limiar de Eficiência do Engajamento")

    prompt_gemini = f"""
    Analise a relação entre o engajamento (IEG) e a ocorrência do 'Ponto de Virada' (PV) de alunos, com base nos seguintes resultados:

    **1. Percentual de Alunos que Atingiram o Ponto de Virada por Faixa de IEG:**
    ```
    {pv_por_ieg_faixa.to_string()}
    ```

    **2. Limiar de Engajamento (IEG) Sugerido:**
    O IEG mediano entre todos os alunos é {ieg_cut_off_value:.2f}.

    Com base nesses dados, por favor, forneça:
    1. Uma interpretação clara do IEG mediano como um limiar e sua importância para o 'Ponto de Virada'.
    2. Quais são as implicações práticas desse limiar para a instituição 'Passos Mágicos' na identificação e suporte de alunos?
    3. Sugira estratégias de intervenção específicas para alunos que se encontram abaixo do IEG mediano, visando aumentar seu engajamento e a probabilidade de atingirem o 'Ponto de Virada'.
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

def display_question_4(df):
    st.header("Pergunta 4: Autoavaliação (IAA)")
    st.markdown("***As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?***")

    df_clean_q4 = df.dropna(subset=["IAA", "IDA", "IEG"]).copy()

    st.markdown("""
    Nesta seção, exploramos a coerência entre a **Autoavaliação (IAA)** dos alunos e seus resultados em **Desempenho Real (IDA)** e **Engajamento (IEG)**. Uma percepção alinhada com a realidade é crucial para o desenvolvimento do aluno.
    """)

    # Coeficientes de correlação (Pearson):
    corr_iaa_ida, p_iaa_ida = pearsonr(df_clean_q4["IAA"], df_clean_q4["IDA"])
    corr_iaa_ieg, p_iaa_ieg = pearsonr(df_clean_q4["IAA"], df_clean_q4["IEG"])

    st.subheader("Análise de Correlação")
    st.write(f"- **Correlação IAA x IDA**: {corr_iaa_ida:.3f}, p = {p_iaa_ida:.3f}")
    st.write(f"- **Correlação IAA x IEG**: {corr_iaa_ieg:.3f}, p = {p_iaa_ieg:.3f}")
    st.markdown(
        "<small><i>Valores perto de 1 indicam forte coerência positiva; perto de 0 indicam pouca relação. "
        "p < 0.05 sugere que a correlação não é devido ao acaso.</i></small>",
        unsafe_allow_html=True
    )
    st.markdown("""
    As correlações de Pearson nos ajudam a quantificar a força e a direção da relação entre a autoavaliação do aluno e seu desempenho e engajamento. Uma correlação positiva indica que, à medida que um aumenta, o outro também tende a aumentar. O valor-p nos informa sobre a significância estatística dessa relação.
    """)

    st.subheader("Visualizações")
    st.markdown("""
    Os gráficos de dispersão a seguir visualizam a relação entre a autoavaliação (IAA) e os outros indicadores. A linha tracejada roxa representa a linha de 45 graus, onde os valores de X e Y são iguais, indicando uma perfeita coerência. A linha de regressão vermelha mostra a tendência geral dos dados.
    """)

    # Gráfico 1: IAA (autoavaliação) vs IDA (desempenho real)
    plot_reg_with_45_deg_line(df_clean_q4, "IDA", "IAA",
                              "Autoavaliação (IAA) vs Desempenho Real (IDA)",
                              "IDA (Desempenho Real)", "IAA (Autoavaliação)")
    st.pyplot(plt.gcf())
    plt.close() # Fecha a figura para evitar sobreposição em futuras chamadas
    st.markdown("""
    **Gráfico de Dispersão (IAA vs IDA)**: Observe como os pontos se distribuem em relação à linha de 45 graus. Alunos cujos pontos estão próximos a essa linha tendem a ter uma autoavaliação coerente com seu desempenho real. Pontos acima da linha podem indicar superestimação (IAA > IDA), enquanto pontos abaixo podem sugerir subestimação (IAA < IDA).
    """)

    # Gráfico 2: IAA (autoavaliação) vs IEG (engajamento)
    plot_reg_with_45_deg_line(df_clean_q4, "IEG", "IAA",
                              "Autoavaliação (IAA) vs Engajamento (IEG)",
                              "IEG (Engajamento)", "IAA (Autoavaliação)")
    st.pyplot(plt.gcf())
    plt.close()
    st.markdown("""
    **Gráfico de Dispersão (IAA vs IEG)**: Similarmente, este gráfico mostra a relação entre a autoavaliação e o engajamento. A linha de 45 graus serve como um benchmark para a coerência. A dispersão dos pontos pode revelar se alunos com alto engajamento tendem a se autoavaliar melhor, ou vice-versa, e se há desalinhamentos significativos.
    """)

    # Gráfico 3: IAA vs IDA por Sexo
    plt.figure(figsize=(10, 7))
    
    g = sns.lmplot(x="IDA", y="IAA", hue="Sexo", data=df_clean_q4, height=6, aspect=1.2,
               markers=["o", "s"], scatter_kws={"s": 60, "alpha": 0.7})
    g.fig.suptitle("Autoavaliação (IAA) vs Desempenho Real (IDA) por Sexo", y=1.02)
    g.set_axis_labels("IDA (Desempenho Real)", "IAA (Autoavaliação)")
    g.add_legend(title="Sexo")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(g.fig)
    plt.close(g.fig)
    st.markdown("""
    **Gráfico de Dispersão (IAA vs IDA por Sexo)**: Este gráfico segmenta a análise da coerência entre autoavaliação e desempenho por gênero. É possível observar se há diferenças nos padrões de superestimação ou subestimação entre sexos, o que pode indicar a necessidade de abordagens pedagógicas ou de suporte psicológico diferenciadas.
    """)

    # --- Matriz de Perfil Psico-Pedagógico (IDA e IAA) ---
    st.subheader("Matriz de Perfil Psico-Pedagógico (IDA vs IAA)")
    st.markdown("Esta análise categoriza os alunos em quadrantes baseados no seu desempenho real (IDA) e autoavaliação (IAA) em relação às medianas do grupo.")

    median_ida = df_clean_q4['IDA'].median()
    median_iaa = df_clean_q4['IAA'].median()

    df_clean_q4['IDA_Cat'] = df_clean_q4['IDA'].apply(lambda x: 'Alta' if x >= median_ida else 'Baixa')
    df_clean_q4['IAA_Cat'] = df_clean_q4['IAA'].apply(lambda x: 'Alta' if x >= median_iaa else 'Baixa')

    def assign_quadrant(row):
        if row['IDA_Cat'] == 'Alta' and row['IAA_Cat'] == 'Alta':
            return 'Consciente' # Bom desempenho, boa auto-percepção
        elif row['IDA_Cat'] == 'Alta' and row['IAA_Cat'] == 'Baixa':
            return 'Inseguro'   # Bom desempenho, mas auto-percepção baixa
        elif row['IDA_Cat'] == 'Baixa' and row['IAA_Cat'] == 'Alta':
            return 'Iludido'    # Baixo desempenho, mas auto-percepção alta
        else: # IDA Baixa and IAA Baixa
            return 'Em Risco'   # Baixo desempenho, baixa auto-percepção

    df_clean_q4['Perfil_Psico_Pedagogico'] = df_clean_q4.apply(assign_quadrant, axis=1)

    quadrant_counts = df_clean_q4['Perfil_Psico_Pedagogico'].value_counts()
    quadrant_percentages = df_clean_q4['Perfil_Psico_Pedagogico'].value_counts(normalize=True) * 100
    quadrant_analysis_df = pd.DataFrame({
        'Contagem de Alunos': quadrant_counts,
        'Percentual (%)': quadrant_percentages.round(2)
    }).sort_values(by='Percentual (%)', ascending=False)

    st.write("**Distribuição de Alunos por Quadrante:**")
    st.dataframe(quadrant_analysis_df)

    fig_quadrant, ax_quadrant = plt.subplots(figsize=(10, 6))
    sns.barplot(x=quadrant_analysis_df.index, y=quadrant_analysis_df['Percentual (%)'], hue=quadrant_analysis_df.index, palette='viridis', ax=ax_quadrant, legend=False)
    ax_quadrant.set_title('Percentual de Alunos por Perfil Psico-Pedagógico')
    ax_quadrant.set_xlabel('Perfil')
    ax_quadrant.set_ylabel('Percentual de Alunos (%)')
    ax_quadrant.tick_params(axis='x', rotation=45)
    ax_quadrant.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_quadrant)
    plt.close(fig_quadrant)
    st.markdown("""
    Este gráfico oferece uma visão visual rápida da proporção de alunos em cada perfil. É uma forma eficaz de identificar rapidamente os grupos predominantes e a magnitude dos desafios (como 'Iludidos' e 'Em Risco') que a instituição pode enfrentar.
    """)

    # --- Geração de Insights com a API do Gemini ---
    
    prompt_gemini = f"""
    Apresente uma análise do perfil psicopedagógico de alunos, divididos em quatro quadrantes com base em seu desempenho real (IDA) e autoavaliação (IAA).
    O objetivo é identificar padrões e sugerir ações estratégicas para aprimorar o suporte aos estudantes.

    **Definições dos Quadrantes:**
    - **Consciente**: Alunos com IDA Alta e IAA Alta. (Bom desempenho e boa auto-percepção)
    - **Inseguro**: Alunos com IDA Alta e IAA Baixa. (Bom desempenho, mas auto-percepção baixa)
    - **Iludido**: Alunos com IDA Baixa e IAA Alta. (Baixo desempenho, mas auto-percepção alta)
    - **Em Risco**: Alunos com IDA Baixa e IAA Baixa. (Baixo desempenho e baixa auto-percepção)

    **Percentuais de Alunos por Quadrante:**
    {quadrant_analysis_df.to_string()}

    Com base nesses dados, por favor, forneça:
    1. Uma análise concisa do perfil predominante dos alunos.
    2. Insights estratégicos específicos e recomendações de ações para cada um dos quatro grupos de alunos, considerando suas características.
    3. Dê especial atenção aos grupos 'Iludido' e 'Em Risco'. Se o grupo 'Iludido' for significativo, discuta as implicações de uma autoavaliação inflacionada em relação ao desempenho real e como abordá-la. Se o grupo 'Em Risco' for significativo, proponha estratégias de intervenção urgentes.
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

def display_question_5(df):
    st.header("Pergunta 5: Aspectos Psicossociais (IPS)")
    st.markdown("***Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?***")
    st.markdown("""
    Nesta seção, investigamos se os **Aspectos Psicossociais (IPS)** dos alunos podem estar relacionados com **quedas no Índice de Desenvolvimento do Aluno (INDE)**. Entender essa relação é fundamental para intervenções proativas que apoiem não apenas o desempenho acadêmico, mas também o bem-estar geral do estudante.
    """)

    df['delta_INDE_22_23'] = df['INDE 2023'] - df['INDE 2022']
    df['delta_INDE_23_24'] = df['INDE 2024'] - df['INDE 2023']

    threshold = -0.3 
    df['queda_22_23'] = df['delta_INDE_22_23'] <= threshold
    df['queda_23_24'] = df['delta_INDE_23_24'] <= threshold

    corr_ips_delta = df[['IPS', 'delta_INDE_22_23', 'delta_INDE_23_24']].corr()
    st.subheader("Correlação (IPS vs variações no INDE):")
    st.dataframe(corr_ips_delta['IPS'])
    st.markdown("""
    Os coeficientes de correlação mostram a força e a direção da relação linear entre o IPS e as variações do INDE. Um valor positivo indica que, conforme o IPS aumenta, a variação do INDE também tende a aumentar (ou seja, menos quedas ou mais ganhos). Um valor negativo sugere o oposto.
    """)

    corr_ips_delta_str = corr_ips_delta['IPS'].to_string()

    results_22_23 = analyze_and_plot_queda_streamlit(df, '22_23')
    results_23_24 = analyze_and_plot_queda_streamlit(df, '23_24')

    # --- Geração de Insights com a API do Gemini ---

    prompt_gemini = f"""
Analise os padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento, com base nos seguintes resultados:

**1. Correlação entre IPS e Variações no INDE:**
```
{corr_ips_delta_str}
```

**2. Análise para a Queda 22_23:**
- **IPS médio (queda 22_23 vs. sem queda):**
```
{results_22_23.get('ips_mean_stats', 'N/A')}
```
- **Teste T de IPS (queda vs. sem queda):** {results_22_23.get('ttest_results', 'N/A')}
- **Regressão Logística (queda_22_23 ~ IPS + covariáveis):**
```
{results_22_23.get('logit_summary', 'N/A')}
```

**3. Análise para a Queda 23_24:**
- **IPS médio (queda 23_24 vs. sem queda):**
```
{results_23_24.get('ips_mean_stats', 'N/A')}
```
- **Teste T de IPS (queda vs. sem queda):** {results_23_24.get('ttest_results', 'N/A')}
- **Regressão Logística (queda_23_24 ~ IPS + covariáveis):**
```
{results_23_24.get('logit_summary', 'N/A')}
```

Com base nesses dados, por favor, forneça:
1. Uma análise concisa de como os aspectos psicossociais (IPS) se relacionam com as quedas no Índice de Desenvolvimento do Aluno (INDE) em diferentes períodos.
2. Identifique quais indicadores psicossociais (se houver, a partir da regressão logística) são mais preditivos de uma queda de desempenho.
3. Sugira ações estratégicas e intervenções práticas que a instituição 'Passos Mágicos' pode implementar para mitigar quedas de desempenho, considerando os insights dos dados.
4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


def display_question_6(df):
    st.header("Pergunta 6: Aspectos Psicopedagógicos (IPP)")
    st.markdown("***As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?***")
    st.markdown("""
    Nesta seção, investigamos se os **Aspectos Psicossociais (IPS)** dos alunos podem estar relacionados com **quedas no Índice de Desenvolvimento do Aluno (INDE)**. Entender essa relação é fundamental para intervenções proativas que apoiem não apenas o desempenho acadêmico, mas também o bem-estar geral do estudante.
    """)

    df_clean_q6 = df.dropna(subset=["IPP", "IAN"]).copy()

    # Correlação de Pearson
    correlation_ipp_ian, p_value_ipp_ian = pearsonr(df_clean_q6['IPP'], df_clean_q6['IAN'])

    st.subheader("Análise de Correlação")
    st.write(f"- **Correlação de Pearson entre IPP e IAN**: {correlation_ipp_ian:.3f}")
    st.write(f"- **Valor-P**: {p_value_ipp_ian:.3f}")
    st.markdown("""
    O coeficiente de correlação de Pearson indica a força e a direção da relação linear entre o IPP e o IAN. Um valor próximo de 1 ou -1 sugere uma forte relação, enquanto um valor próximo de 0 indica uma relação fraca. O valor-P ajuda a determinar a significância estatística dessa correlação.
    """)

    median_ian = df_clean_q6['IAN'].median()
    st.write(f"- **Mediana do IAN**: {median_ian:.2f}")

    df_clean_q6['IAN_category'] = df_clean_q6['IAN'].apply(lambda x: 'Baixo IAN' if x <= median_ian else 'Alto IAN')
    st.subheader("Categorização do IAN")
    st.dataframe(df_clean_q6[['IAN', 'IAN_category']].head())
    st.markdown("""
    Dividimos os alunos em 'Baixo IAN' e 'Alto IAN' com base na mediana. Isso nos permite comparar as avaliações psicopedagógicas (IPP) entre grupos com diferentes níveis de atraso de desenvolvimento, facilitando a identificação de padrões.
    """)

    ipp_low_ian = df_clean_q6.loc[df_clean_q6['IAN_category'] == 'Baixo IAN', 'IPP']
    ipp_high_ian = df_clean_q6.loc[df_clean_q6['IAN_category'] == 'Alto IAN', 'IPP']

    t_statistic, p_value_ttest, mean_ipp_low_ian, mean_ipp_high_ian = None, None, None, None

    if len(ipp_low_ian) > 1 and len(ipp_high_ian) > 1:
        t_statistic, p_value_ttest = ttest_ind(ipp_low_ian, ipp_high_ian, equal_var=False)
        st.subheader("Teste T comparando IPP entre 'Baixo IAN' e 'Alto IAN':")
        st.write(f"- **Estatística T**: {t_statistic:.3f}")
        st.write(f"- **Valor-P**: {p_value_ttest:.3f}")

        mean_ipp_low_ian = ipp_low_ian.mean()
        mean_ipp_high_ian = ipp_high_ian.mean()

        st.write(f"- **IPP Médio para o grupo 'Baixo IAN'**: {mean_ipp_low_ian:.3f}")
        st.write(f"- **IPP Médio para o grupo 'Alto IAN'**: {mean_ipp_high_ian:.3f}")
        st.markdown("""
        O Teste T nos diz se a diferença nas médias do IPP entre os grupos 'Baixo IAN' e 'Alto IAN' é estatisticamente significativa. Um valor-P baixo (geralmente < 0.05) sugere que existe uma diferença real, não apenas uma flutuação aleatória, confirmando ou contradizendo a relação esperada entre IPP e IAN.
        """)
    else:
        st.write("Não há dados suficientes para realizar o Teste T.")

    st.subheader("Visualizações")

    # Gráfico de Dispersão (Scatter Plot) de IPP vs IAN
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_clean_q6, x='IAN', y='IPP', alpha=0.7, ax=ax1)
    ax1.set_title('Dispersão de IPP vs IAN')
    ax1.set_xlabel('IAN (Índice de Atraso de Desenvolvimento)')
    ax1.set_ylabel('IPP (Índice de Perfil Psicopedagógico)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)
    plt.close(fig1)
    st.markdown("""
    Este gráfico de dispersão visualiza a relação direta entre as avaliações psicopedagógicas (IPP) e o índice de atraso de desenvolvimento (IAN). Podemos observar a tendência dos pontos: se alunos com IAN mais alto tendem a ter IPP mais baixos (ou vice-versa), indicando uma coerência entre as métricas.
    """)

    # Box Plot de IPP por Categoria de IAN
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_clean_q6, x='IAN_category', y='IPP', hue='IAN_category', palette='pastel', legend=False, ax=ax2)
    ax2.set_title('Box Plot de IPP por Categoria de IAN')
    ax2.set_xlabel('Categoria de IAN')
    ax2.set_ylabel('IPP (Índice de Perfil Psicopedagógico)')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown("""
    O gráfico compara a distribuição do IPP para alunos categorizados com 'Baixo IAN' e 'Alto IAN'. As caixas mostram a mediana, quartis e possíveis outliers, permitindo uma clara visualização das diferenças no perfil psicopedagógico entre os grupos com menor e maior atraso de desenvolvimento.
    """)

    # --- Geração de Insights com a API do Gemini ---

    prompt_gemini = f"""
    Analise se as avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN (Índice de Atraso de Desenvolvimento), com base nos seguintes resultados:

    **Resultados da Análise:**
    - Correlação de Pearson entre IPP e IAN: {correlation_ipp_ian:.3f} (p-valor: {p_value_ipp_ian:.3f})
    - Mediana do IAN utilizada para categorização: {median_ian:.2f}
    """

    if t_statistic is not None and p_value_ttest is not None:
        prompt_gemini += f"""
    - Teste T de amostras independentes comparando IPP entre 'Baixo IAN' e 'Alto IAN':
      - Estatística T: {t_statistic:.3f}
      - Valor-P: {p_value_ttest:.3f}
      - IPP Médio para 'Baixo IAN': {mean_ipp_low_ian:.3f}
      - IPP Médio para 'Alto IAN': {mean_ipp_high_ian:.3f}
    """
    else:
        prompt_gemini += "\n    - Não foi possível realizar o Teste T devido a dados insuficientes.\n"

    prompt_gemini += f"""

    Com base nesses dados, por favor, forneça:
    1. Uma conclusão clara sobre se o IPP confirma ou contradiz a defasagem identificada pelo IAN.
    2. Quais são as implicações práticas dessa relação para a "Passos Mágicos"?
    3. Sugira ações estratégicas ou programas que a instituição pode implementar para abordar as defasagens psicopedagógicas, considerando a relação com o IAN.
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

def display_question_7(df):
    st.header("Pergunta 7: Ponto de Virada")
    st.markdown("***Quais comportamentos - acadêmicos, emocionais ou de engajamento - mais influenciam o IPV ao longo do tempo?***")

    required_cols_q7 = ["IDA", "IEG", "IPS", "IAA", "IPP", "N_Atingiu PV"]
    df_q7 = df.dropna(subset=required_cols_q7).copy()

    if df_q7.empty:
        st.warning("Não há dados suficientes para realizar a análise da Pergunta 7 após a remoção de valores ausentes.")
        return

    X = df_q7[["IDA", "IEG", "IPS", "IAA", "IPP"]]
    y = df_q7["N_Atingiu PV"]

    if y.nunique() < 2:
        st.warning("A variável alvo 'N_Atingiu PV' tem menos de 2 classes únicas, o que impede a criação do modelo de classificação.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_pv = RandomForestClassifier(random_state=42)
    model_pv.fit(X_train, y_train)

    prob_pv = model_pv.predict_proba(X_test)[:, 1]
    auc_pv = roc_auc_score(y_test, prob_pv)

    st.subheader("Desempenho do Modelo Preditivo")
    st.metric("AUC do Modelo (Ponto de Virada)", f"{auc_pv:.3f}")

    importances = pd.DataFrame({
        "Indicador": X.columns,
        "Importancia": model_pv.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    fig_importances, ax_importances = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importancia", y="Indicador", data=importances, palette='viridis', ax=ax_importances)
    ax_importances.set_title("Importância das Variáveis no Ponto de Virada")
    ax_importances.set_xlabel("Peso no Modelo")
    st.pyplot(fig_importances)
    plt.close(fig_importances)

    st.write(f"""O modelo preditivo desenvolvido para identificar o “Ponto de Virada” apresentou AUC de {auc_pv:.3f}, o que indica excelente capacidade de discriminação.""")
    st.write("Um AUC próximo de 0,90 demonstra que o modelo consegue diferenciar com alta precisão os alunos que atingem o ponto de virada daqueles que não atingem. O gráfico de importância das variáveis mostra quais indicadores possuem maior peso na previsão, permitindo identificar os fatores mais determinantes para mudança significativa no desempenho.")
    st.write("Esse resultado valida a utilização do modelo como ferramenta estratégica para tomada de decisão.")
    
    st.subheader("Simulador Interativo do Ponto de Virada")
    st.markdown("Ajuste os indicadores abaixo para ver como eles influenciam a probabilidade de um aluno atingir o 'Ponto de Virada'.")

    # Sliders para os indicadores com faixa de 0 a 10
    ida_sim = st.slider("IDA (Desempenho Acadêmico)", min_value=0.0, max_value=10.0, value=float(df_q7['IDA'].mean()))
    ieg_sim = st.slider("IEG (Engajamento)", min_value=0.0, max_value=10.0, value=float(df_q7['IEG'].mean()))
    ips_sim = st.slider("IPS (Aspectos Psicossociais)", min_value=0.0, max_value=10.0, value=float(df_q7['IPS'].mean()))
    iaa_sim = st.slider("IAA (Autoavaliação)", min_value=0.0, max_value=10.0, value=float(df_q7['IAA'].mean()))
    ipp_sim = st.slider("IPP (Perfil Psicopedagógico)", min_value=0.0, max_value=10.0, value=float(df_q7['IPP'].mean()))

    simulated_data = pd.DataFrame([{
        "IDA": ida_sim,
        "IEG": ieg_sim,
        "IPS": ips_sim,
        "IAA": iaa_sim,
        "IPP": ipp_sim
    }])

    simulated_prob = model_pv.predict_proba(simulated_data)[:, 1][0]
    st.metric("Probabilidade de Atingir o Ponto de Virada", f"{simulated_prob:.2%}")

    # --- Geração de Insights com a API do Gemini para Plano de Metas Individualizado ---
    st.subheader("Plano de Metas Individualizado")

    if st.button("Gerar Plano de Metas para Simulação"):
        prompt_gemini = f"""
        Um aluno tem os seguintes indicadores:
        - IDA (Desempenho Acadêmico): {ida_sim:.2f}
        - IEG (Engajamento): {ieg_sim:.2f}
        - IPS (Aspectos Psicossociais): {ips_sim:.2f}
        - IAA (Autoavaliação): {iaa_sim:.2f}
        - IPP (Perfil Psicopedagógico): {ipp_sim:.2f}
        Todos os indicadores (IDA, IEG, IPS, IAA, IPP) são medidos em uma escala de 0 a 10.
        
        A probabilidade atual estimada de ele atingir o 'Ponto de Virada' é de {simulated_prob:.2%}.

        Com base nesses dados, gere um 'Plano de Metas Individualizado' para este aluno, visando aumentar sua probabilidade de atingir o Ponto de Virada. O plano deve ser prático, com sugestões claras e acionáveis, e focado nos indicadores que podem ser melhorados. Considere os impactos de cada indicador na probabilidade. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
        """

        gemini_insights = generate_gemini_insights(prompt_gemini)

        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar o plano de metas com o Gemini. Verifique a configuração da API.")

def display_question_8(df):
    st.header("Pergunta 8: Multidimensionalidade dos Indicadores")
    st.markdown("***Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a nota global do aluno (INDE)?***")

    df_regression_q8 = df.copy()

    columns_for_regression = ['INDE 2023', 'IDA', 'IEG', 'IPS', 'IPP']
    df_regression_q8.dropna(subset=columns_for_regression, inplace=True)

    st.subheader("Informações do DataFrame para Regressão")
    st.write("5 primeiras linhas do DataFrame limpo para regressão:")
    st.dataframe(df_regression_q8[columns_for_regression].head())

    # Modelo de Regressão Linear (OLS)
    model_formula = 'Q("INDE 2023") ~ Q("IDA") + Q("IEG") + Q("IPS") + Q("IPP")'
    regression_model = smf.ols(model_formula, data=df_regression_q8).fit()
    st.subheader("Sumário do Modelo de Regressão OLS:")
    
    results_df = pd.DataFrame({
        'Variável': regression_model.params.index,
        'Coeficiente': regression_model.params.values,
        'P-valor': regression_model.pvalues.values
    })
    results_df['Coeficiente'] = results_df['Coeficiente'].map('{:,.3f}'.format)
    results_df['P-valor'] = results_df['P-valor'].map('{:,.3f}'.format)
    st.dataframe(results_df)
    
    st.write(f"- **R-quadrado**: {regression_model.rsquared:.3f}")
    st.write(f"- **R-quadrado Ajustado**: {regression_model.rsquared_adj:.3f}")

    ols_summary_text = regression_model.summary().as_text() # Capturar para o prompt, se necessário
    st.markdown("""
    Este sumário apresenta os resultados da regressão linear. O **R-quadrado** indica a proporção da variância do INDE 2023 que é explicada pelos indicadores. Os **coeficientes (coef)** mostram a direção e a magnitude do impacto de cada indicador no INDE. Um coeficiente positivo significa que o aumento do indicador está associado ao aumento do INDE, e vice-versa. O **P>|t|** (p-valor) indica a significância estatística de cada coeficiente; valores menores que 0.05 geralmente sugerem que o indicador tem um efeito significativo no INDE.
    """)

    q1_inde_2023 = df_regression_q8['INDE 2023'].quantile(0.25)
    q3_inde_2023 = df_regression_q8['INDE 2023'].quantile(0.75)

    st.subheader("Análise de Percentis do INDE 2023")
    st.write(f"- **25º percentil (Q1) do INDE 2023**: {q1_inde_2023:.2f}")
    st.write(f"- **75º percentil (Q3) do INDE 2023**: {q3_inde_2023:.2f}")

    st.markdown("""
    Utilizamos o 25º e 75º percentis do INDE 2023 para categorizar os alunos, respectivamente, em grupos de 'Baixo INDE' e 'Alto INDE'. Essa segmentação nos permite comparar os perfis de indicadores de alunos com desempenho inferior e superior.
    """)

    low_inde_students = df_regression_q8[df_regression_q8['INDE 2023'] <= q1_inde_2023].copy()
    high_inde_students = df_regression_q8[df_regression_q8['INDE 2023'] >= q3_inde_2023].copy()

    st.write("5 primeiras linhas de alunos com 'Baixo INDE':")
    st.dataframe(low_inde_students[['INDE 2023']].head())
    st.write("5 primeiras linhas de alunos com 'Alto INDE':")
    st.dataframe(high_inde_students[['INDE 2023']].head())

    indicator_cols = ['IDA', 'IEG', 'IPS', 'IPP']

    scaler = MinMaxScaler()
    scaler.fit(df_regression_q8[indicator_cols])

    low_inde_students_normalized = low_inde_students.copy()
    high_inde_students_normalized = high_inde_students.copy()

    low_inde_students_normalized[indicator_cols] = scaler.transform(low_inde_students_normalized[indicator_cols])
    high_inde_students_normalized[indicator_cols] = scaler.transform(high_inde_students_normalized[indicator_cols])

    avg_low_inde = low_inde_students_normalized[indicator_cols].mean()
    avg_high_inde = high_inde_students_normalized[indicator_cols].mean()

    comparison_df = pd.DataFrame({
        'Baixo INDE': avg_low_inde,
        'Alto INDE': avg_high_inde
    })

    st.subheader("Médias Normalizadas dos Indicadores para grupos 'Baixo INDE' vs 'Alto INDE':")
    st.dataframe(comparison_df)
    comparison_df_str = comparison_df.to_string()
    st.markdown("""
    Esta tabela exibe as médias normalizadas dos indicadores para os grupos de 'Baixo INDE' e 'Alto INDE'. A normalização (escala de 0 a 1) permite uma comparação justa entre indicadores com diferentes escalas originais. Podemos observar quais indicadores são consistentemente mais altos no grupo de 'Alto INDE', indicando áreas de força para o sucesso acadêmico.
    """)

    # Gráfico de Radar
    indicators = comparison_df.index.tolist()
    low_inde_scores = comparison_df['Baixo INDE'].tolist()
    high_inde_scores = comparison_df['Alto INDE'].tolist()

    low_inde_scores = low_inde_scores + low_inde_scores[:1]
    high_inde_scores = high_inde_scores + high_inde_scores[:1]
    indicators_plot = indicators + indicators[:1]

    angles = np.linspace(0, 2 * np.pi, len(indicators_plot), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, low_inde_scores, color='blue', linewidth=2, label='Baixo INDE')
    ax.fill(angles, low_inde_scores, color='blue', alpha=0.25)

    ax.plot(angles, high_inde_scores, color='red', linewidth=2, label='Alto INDE')
    ax.fill(angles, high_inde_scores, color='red', alpha=0.25)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators)

    ax.set_title('Perfil dos Indicadores para Grupos de Alto e Baixo INDE', va='bottom', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    O gráfico de radar visualiza o perfil médio dos indicadores para os alunos de 'Baixo INDE' e 'Alto INDE'. As áreas sombreadas permitem uma comparação rápida das forças e fraquezas de cada grupo em relação aos diferentes indicadores. Quanto maior a área coberta por um grupo, maior sua pontuação média nos indicadores correspondentes. Isso ajuda a identificar visualmente quais indicadores precisam de maior atenção para elevar o desempenho dos alunos com 'Baixo INDE'.
    """)

    # --- Geração de Insights com a API do Gemini ---

    prompt_gemini = f"""
    Analise a relação multidimensional entre indicadores de desempenho (IDA), engajamento (IEG), aspectos psicossociais (IPS) e psicopedagógicos (IPP) com a nota global do aluno (INDE), com base nos seguintes resultados:

    **1. Sumário do Modelo de Regressão Linear (OLS) para INDE 2023:**
    ```
    {ols_summary_text}
    ```

    **2. Médias Normalizadas dos Indicadores para grupos 'Baixo INDE' (25% inferiores) vs 'Alto INDE' (25% superiores):**
    ```
    {comparison_df_str}
    ```

    Com base nesses dados, por favor, forneça:
    1. Uma interpretação dos coeficientes do modelo OLS, identificando quais indicadores têm o maior impacto positivo ou negativo no INDE.
    2. Uma comparação clara entre os perfis de indicadores de alunos com 'Baixo INDE' e 'Alto INDE'. Quais indicadores se destacam como pontos fortes para alunos de alto desempenho e pontos fracos para alunos de baixo desempenho?
    3. Sugira combinações de indicadores que, se aprimoradas, teriam o maior potencial para elevar a nota global (INDE) dos alunos da 'Passos Mágicos'.
    4. Proponha ações estratégicas e intervenções focadas nos indicadores-chave para melhorar o desempenho geral dos alunos.
    5. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

def display_question_9(df):
    st.header("Pergunta 9: Previsão de risco com Machine Learning")
    st.markdown("***Quais padrões nos indicadores permitem identificar alunos em risco antes de queda no desempenho ou aumento da defasagem? Construa um modelo preditivo que mostre uma probabilidade do aluno ou aluna entrar em risco de defasagem.***")

    required_cols_q9 = ["IDA", "IEG", "IPS", "IAA", "IPP", "Defasagem"]
    df_q9 = df.dropna(subset=required_cols_q9).copy()

    if df_q9.empty:
        st.warning("Não há dados suficientes para realizar a análise da Pergunta 9 após a remoção de valores ausentes.")
        return

    df_q9["Em_Risco"] = np.where(
        df_q9["Defasagem"] > df_q9["Defasagem"].median(), # Assumindo que defasagem maior que a mediana é 'Em Risco'
        1,
        0
    )

    X_risco = df_q9[["IDA", "IEG", "IPS", "IAA", "IPP"]]
    y_risco = df_q9["Em_Risco"]

    if y_risco.nunique() < 2:
        st.warning("A variável alvo 'Em_Risco' tem menos de 2 classes únicas, o que impede a criação do modelo de classificação.")
        return

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_risco, y_risco, test_size=0.3, random_state=42
    )

    model_risco = RandomForestClassifier(random_state=42)
    model_risco.fit(X_train_r, y_train_r)

    prob_risco = model_risco.predict_proba(X_test_r)[:, 1]
    auc_risco = roc_auc_score(y_test_r, prob_risco)

    st.subheader("Desempenho do Modelo de Risco")
    st.metric("AUC do Modelo de Risco", f"{auc_risco:.3f}")

    # --- Curva ROC ---
    fpr, tpr, _ = roc_curve(y_test_r, prob_risco)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_title("Curva ROC - Modelo de Risco")
    ax_roc.set_xlabel("Falso Positivo")
    ax_roc.set_ylabel("Verdadeiro Positivo")
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # --- Distribuição de Probabilidade ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(prob_risco, bins=20)
    ax_hist.set_title("Distribuição da Probabilidade de Risco")
    ax_hist.set_xlabel("Probabilidade")
    ax_hist.set_ylabel("Quantidade")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.write(f"""O modelo de previsão de risco acadêmico apresentou AUC de {auc_risco:.3f}""")
    st.write("Esse valor indica capacidade preditiva moderada. O modelo consegue identificar tendências de risco, porém com menor precisão quando comparado ao modelo de ponto de virada.")
    st.write("Isso sugere que a variável 'defasagem' pode depender de fatores adicionais que não estão totalmente capturados pelos indicadores analisados.")
    st.write("Ainda assim, o modelo é útil como ferramenta de apoio para identificação preventiva de alunos com maior probabilidade de risco.")

    # --- SHAP para Explicabilidade ---
    st.subheader("Explicabilidade do Modelo (SHAP)")
    st.markdown("O SHAP (SHapley Additive exPlanations) ajuda a entender como cada característica individual influencia a previsão do modelo para um aluno específico.")

    explainer = shap.TreeExplainer(model_risco)
    shap_values_raw = explainer.shap_values(X_risco.values)

    # Debug
    # st.write(f"Debug: Type of shap_values_raw: {type(shap_values_raw)}")
    # if isinstance(shap_values_raw, list):
    #     st.write(f"Debug: Length of shap_values_raw: {len(shap_values_raw)}")
    #     if len(shap_values_raw) > 1:
    #         st.write(f"Debug: Shape of shap_values_raw[1]: {shap_values_raw[1].shape}")
    #     else:
    #         st.write(f"Debug: shap_values_raw é uma lista mas tem menos de 2 elementos. Shape de shap_values_raw[0]: {shap_values_raw[0].shape}")
    # else:
    #     st.write(f"Debug: shap_values_raw é um único array. Shape: {shap_values_raw.shape}")
    # st.write(f"Debug: Shape de X_risco: {X_risco.shape}")

    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
        shap_values_for_positive_class = shap_values_raw[1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
        shap_values_for_positive_class = shap_values_raw[:, :, 1]
    else:
        st.error("Erro: `explainer.shap_values` retornou um formato inesperado. Não foi possível extrair os valores SHAP para a classe positiva.")
        return

    if shap_values_for_positive_class.shape != X_risco.shape:
        st.error(f"Erro: A forma dos valores SHAP para a classe positiva ({shap_values_for_positive_class.shape}) não corresponde à forma de X_risco ({X_risco.shape}).")
        st.error("Isso indica um problema na geração dos valores SHAP. Verifique a versão do SHAP e a compatibilidade do modelo.")
        return

    shap_values_df = pd.DataFrame(shap_values_for_positive_class, columns=X_risco.columns, index=X_risco.index)

    df_q9['Prob_Risco'] = model_risco.predict_proba(X_risco)[:, 1]
    fila_prioridade = df_q9.sort_values(by='Prob_Risco', ascending=False).head(10)

    st.write("**Fila de Prioridade (Top 10 Alunos com Maior Probabilidade de Risco):**")
    st.dataframe(fila_prioridade[['RA', 'Fase', 'IDA', 'IEG', 'IPS', 'IAA', 'IPP', 'Prob_Risco']])

    if not fila_prioridade.empty:
        selected_ra = st.selectbox("Selecione um RA da fila de prioridade para ver os fatores contribuintes:", fila_prioridade['RA'].unique())
        if selected_ra:
            student_original_label = df_q9[df_q9['RA'] == selected_ra].index[0]
            st.write(f"**Fatores Contribuintes para o aluno: {selected_ra}**")

            if student_original_label not in shap_values_df.index:
                st.warning(f"O RA {selected_ra} (índice original {student_original_label}) não foi encontrado no DataFrame de valores SHAP. Pode haver uma inconsistência nos dados ou na indexação.")
                return

            shap_values_single = shap_values_df.loc[student_original_label].values

            features_single = X_risco.loc[student_original_label]

            shap_df = pd.DataFrame({
                'Feature': X_risco.columns,
                'SHAP_Value': shap_values_single
            }).sort_values(by='SHAP_Value', key=abs, ascending=False)
            st.dataframe(shap_df)

            fig_shap = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap.Explanation(
                values=shap_values_single,
                base_values=explainer.expected_value[1],
                data=features_single.values,
                feature_names=X_risco.columns.tolist()
            ), show=False)
            st.pyplot(fig_shap)
            plt.close(fig_shap)
    else:
        st.info("A fila de prioridade está vazia para seleção.")

    # --- Geração de Insights com a API do Gemini ---
    # st.subheader("Insights e Recomendações para Alunos de Alto Risco (via Gemini)")

    prompt_gemini = f"""
    Analise o modelo de risco de defasagem, a fila de prioridade de alunos e os fatores contribuintes (SHAP values) para os top alunos em risco.
    
    **Desempenho do Modelo:**
    - AUC do Modelo de Risco: {auc_risco:.3f}

    **Fila de Prioridade (Top 10 Alunos):**
    ```
    {fila_prioridade[['RA', 'Prob_Risco', 'IDA', 'IEG', 'IPS', 'IAA', 'IPP']].to_string()}
    ```

    Com base nesses dados e no conceito de fatores contribuintes do SHAP, forneça:
    1. Uma interpretação do desempenho do modelo (AUC).
    2. Análise dos padrões dos alunos na fila de prioridade e os principais indicadores que os colocam em risco.
    3. Sugestões de ações práticas e personalizadas para a 'Passos Mágicos' intervir com esses alunos de alto risco, focando nos indicadores que mais contribuem para o risco de defasagem (como indicado pelos SHAP values).
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights com o Gemini. Verifique a configuração da API.")

def display_question_10(df):
    st.header("Pergunta 10: Efetividade do programa")
    st.markdown("***Os indicadores mostram melhora consistente ao longo do ciclo nas diferentes fases (Quartzo, Ágata, Ametista e Topázio), confirmando o impacto real do programa?***")

    inde_long = df.melt(
        id_vars=["RA", "Fase"],
        value_vars=["INDE 2022", "INDE 2023"],
        var_name="Ano",
        value_name="INDE"
    )

    inde_long["Ano"] = inde_long["Ano"].str.replace("INDE ", "").astype(int)
    inde_long["Fase"] = pd.to_numeric(inde_long["Fase"], errors="coerce")

    inde_long = inde_long.dropna(subset=["Fase"])

    # --- Evolução do INDE por Fase ---
    st.subheader("Evolução do INDE ao Longo do Tempo por Fase")

    media_fase_ano = (
        inde_long
        .groupby(["Fase", "Ano"])[ "INDE"]
        .mean()
        .reset_index()
    )

    fig_fase_evol, ax_fase_evol = plt.subplots(figsize=(10, 6))
    cores = plt.cm.tab10.colors

    for i, fase in enumerate(sorted(media_fase_ano["Fase"].unique())):
        dados_fase = media_fase_ano[media_fase_ano["Fase"] == fase]
        ax_fase_evol.plot(
            dados_fase["Ano"],
            dados_fase["INDE"],
            marker="o",
            linewidth=2,
            color=cores[i % len(cores)],
            label=f"Fase {int(fase)}"
        )

    ax_fase_evol.set_title("Evolução do INDE ao Longo do Tempo por Fase", fontsize=14, weight="bold")
    ax_fase_evol.set_xlabel("Ano")
    ax_fase_evol.set_ylabel("Média INDE")
    ax_fase_evol.set_xticks(sorted(media_fase_ano["Ano"].unique()))
    ax_fase_evol.legend(title="Fase", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_fase_evol.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_fase_evol)
    plt.close(fig_fase_evol)

    st.write("Observa-se que a maioria das fases apresenta crescimento ao longo do tempo, indicando evolução positiva do desempenho acadêmico.")
    st.write("Algumas fases demonstram crescimento mais acentuado, enquanto outras apresentam evolução mais gradual. Esse comportamento permite identificar quais grupos tiveram maior progresso e quais podem demandar estratégias específicas de acompanhamento.")
    st.write("A análise temporal por fase é fundamental para monitoramento contínuo e planejamento pedagógico estratégico.")

    # --- Cálculo do Índice de Valor Adicionado (IVA) ---
    st.subheader("Cálculo do Índice de Valor Adicionado (IVA)")
    st.markdown("O IVA mede o ganho de desempenho de um aluno em relação à média de sua fase.")

    df_q10 = df.copy()
    df_q10['Evolucao_Aluno'] = df_q10['INDE 2023'] - df_q10['INDE 2022']

    media_evolucao_fase = df_q10.groupby('Fase')['Evolucao_Aluno'].mean().reset_index()
    media_evolucao_fase.rename(columns={'Evolucao_Aluno': 'Media_Evolucao_da_Fase'}, inplace=True)

    df_q10 = pd.merge(df_q10, media_evolucao_fase, on='Fase', how='left')

    df_q10['IVA'] = df_q10['Evolucao_Aluno'] - df_q10['Media_Evolucao_da_Fase']

    st.write("**Top 10 Alunos com Maior Valor Adicionado (IVA):**")
    st.dataframe(df_q10.sort_values(by='IVA', ascending=False).head(10)[['RA', 'Fase', 'Evolucao_Aluno', 'Media_Evolucao_da_Fase', 'IVA']])

    st.write("**Top 10 Alunos com Menor Valor Adicionado (IVA):**")
    st.dataframe(df_q10.sort_values(by='IVA', ascending=True).head(10)[['RA', 'Fase', 'Evolucao_Aluno', 'Media_Evolucao_da_Fase', 'IVA']])

    # --- Selo de Impacto (Agregado) ---
    st.subheader("Selo de Impacto da Passos Mágicos")
    st.markdown("Avaliação agregada do impacto da Passos Mágicos no desempenho dos alunos, ideal para atrair investidores.")

    # Calcular a média geral do IVA
    media_iva_geral = df_q10['IVA'].mean()
    st.metric("Média Geral do Valor Adicionado (IVA)", f"{media_iva_geral:.2f}")

    st.info("Um IVA positivo indica que, em média, os alunos da Passos Mágicos superam a evolução esperada para sua fase, demonstrando o valor adicionado pela instituição.")

    st.markdown("""
    **Entendendo a Média Geral do IVA:**
    *   O Índice de Valor Adicionado (IVA) é calculado pela diferença entre a evolução individual de um aluno no INDE e a evolução média de sua respectiva Fase.
    *   Um IVA **positivo** indica que o aluno superou a média de evolução de sua fase. Um IVA **negativo** indica que o aluno evoluiu menos que a média de sua fase.
    *   Uma **Média Geral do IVA igual a 0.00** significa que, em média, o impacto líquido da \"Passos Mágicos\" sobre a evolução dos alunos (considerando o desempenho em relação às suas fases) foi neutro neste período. Ou seja, os ganhos de alguns alunos que superaram a média foram compensados por outros que ficaram abaixo, resultando em um equilíbrio.
    *   Um IVA geral **próximo de zero** é um resultado matematicamente correto e sugere que, embora a instituição possa estar mantendo os alunos no ritmo geral de suas fases, há espaço para otimizar as intervenções e impulsionar a maioria dos alunos a superar consistentemente a evolução esperada.
    *   Para atrair investidores, um IVA positivo e crescente seria mais desejável, pois demonstraria que a instituição está elevando o patamar de desempenho de seus alunos acima do que seria esperado organicamente.
    """)

    # --- Geração de Insights com a API do Gemini para Investidores ---
    # st.subheader("Insights do Gemini para Investidores")

    prompt_gemini = f"""
    Prepare um resumo executivo para potenciais investidores, destacando o "Valor Adicionado" que a "Passos Mágicos" gera no desempenho dos alunos. Utilize os seguintes dados:

    **1. Média Geral do Índice de Valor Adicionado (IVA):** {media_iva_geral:.2f}

    **2. Evolução do INDE ao Longo do Tempo por Fase (tendências gerais):
       - Média INDE 2022: {media_fase_ano[media_fase_ano['Ano'] == 2022]['INDE'].mean():.2f}
       - Média INDE 2023: {media_fase_ano[media_fase_ano['Ano'] == 2023]['INDE'].mean():.2f}

    **3. Impacto do IVA:**
       - Top 10 Alunos com Maior IVA:
        ```
        {df_q10.sort_values(by='IVA', ascending=False).head(10)[['RA', 'Fase', 'IVA']].to_string()}
        ```

    **Considerações sobre a Média Geral do IVA (para contextualização):**
    A Média Geral do IVA é 0.00 (ou próxima de zero). Isso indica que, em média, o impacto líquido da \"Passos Mágicos\" sobre a evolução dos alunos foi neutro em relação à evolução esperada para suas fases. Apresente essa informação de forma estratégica para investidores, focando no potencial de crescimento e nas áreas onde a instituição já demonstrou sucesso.

    Com base nesses dados, apresente:
    1. Uma narrativa convincente sobre o impacto educacional da Passos Mágicos, focando no IVA como prova de eficácia.
    2. Pontos chave sobre a evolução do INDE que demonstrem sucesso e consistência.
    3. Argumentos que justifiquem o investimento na Passos Mágicos, baseados no valor agregado que a instituição proporciona aos alunos.
    4. Formule a resposta de forma clara, concisa e orientada para investidores, utilizando linguagem persuasiva e dados para corroborar as afirmações.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights para investidores com o Gemini. Verifique a configuração da API.")

def display_question_11(df, numeric_cols_to_clean):
    st.header("Pergunta 11: Insights Adicionais e Criatividade")
    st.markdown("***Utilizando cruzamentos de dados não solicitados nas perguntas anteriores para gerar sugestões práticas que melhorem a operação da instituição.***")

    df['INDE_Consolidado'] = df[numeric_cols_to_clean[-3:]].mean(axis=1, skipna=True)

    st.subheader("Análise do INDE Consolidado por Tipo de Escola")
    st.write("5 primeiras linhas com 'INDE 2022', 'INDE 2023', 'INDE 2024' e 'INDE_Consolidado':")
    st.dataframe(df[numeric_cols_to_clean[-3:] + ['INDE_Consolidado']].head())
    st.markdown("""
    O INDE Consolidado é a média dos índices de desempenho dos anos 2022, 2023 e 2024. Esta consolidação oferece uma visão mais estável e abrangente do desempenho geral do aluno ao longo do tempo.
    """)

    df_clean_school = df.copy()
    df_clean_school['IE'] = df_clean_school['IE'].astype(str).str.strip().str.lower()

    def clean_school_type(school_type):
        if 'pública' in school_type:
            return 'Pública'
        elif 'privada' in school_type or 'rede decisão' in school_type:
            return 'Privada'
        else:
            return 'Outros'

    df_clean_school['IE_Limpo'] = df_clean_school['IE'].apply(clean_school_type)

    st.write("Valores únicos na coluna 'IE_Limpo' e suas contagens após a limpeza:")
    st.dataframe(df_clean_school['IE_Limpo'].value_counts())
    ie_counts_str = df_clean_school['IE_Limpo'].value_counts().to_string()
    st.markdown("""
    A limpeza e padronização da coluna 'IE' (Instituição de Ensino) permite agrupar e analisar o desempenho dos alunos por tipo de escola, como 'Pública' ou 'Privada'. Isso revela a distribuição da nossa base de alunos entre os diferentes tipos de instituições e pode indicar se a 'Passos Mágicos' atende majoritariamente a um tipo específico de escola.
    """)

    school_inde_stats = df_clean_school.groupby('IE_Limpo')['INDE_Consolidado'].agg(['mean', 'std', 'count'])

    st.subheader("Estatísticas Agregadas do INDE Consolidado por Tipo de Escola:")
    st.dataframe(school_inde_stats)
    school_inde_stats_str = school_inde_stats.to_string()
    st.markdown("""
    Esta tabela detalha a média, desvio padrão e contagem de alunos do INDE Consolidado para cada tipo de escola. Diferenças nas médias podem apontar para a necessidade de abordagens distintas ou revelar onde a 'Passos Mágicos' tem maior ou menor impacto.
    """)

    # Gráfico de Barras: Média do INDE Consolidado por Tipo de Escola
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(x=school_inde_stats.index, y=school_inde_stats['mean'], hue=school_inde_stats.index, palette='viridis', legend=False, ax=ax_bar)
    ax_bar.set_title('Média do INDE Consolidado por Tipo de Escola', fontsize=16)
    ax_bar.set_xlabel('Tipo de Escola', fontsize=12)
    ax_bar.set_ylabel('Média do INDE Consolidado', fontsize=12)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    ax_bar.tick_params(axis='x', labelsize=10)
    ax_bar.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)
    st.markdown("""
    A visualização gráfica da média do INDE Consolidado por tipo de escola facilita a comparação visual do desempenho entre os alunos de escolas públicas, privadas e outros. Pode-se inferir se há um tipo de escola onde os alunos, em média, apresentam um INDE mais alto ou mais baixo, o que pode direcionar estratégias de engajamento ou apoio.
    """)

    st.subheader("Análise de Palavras-Chave nos Destaques")
    df_keywords = df.copy()

    columns_for_keywords_expected = ['Destaque IEG', 'Destaque IDA']
    existing_highlight_columns = [col for col in columns_for_keywords_expected if col in df_keywords.columns]

    if not existing_highlight_columns:
        st.write(f"Aviso: Nenhuma das colunas de destaque esperadas ({', '.join(existing_highlight_columns)}) foi encontrada. 'Destaques Combinados' será uma coluna vazia.")
        df_keywords['Destaques_Combinados'] = ''
    else:
        for col in existing_highlight_columns:
            df_keywords[col] = df_keywords[col].astype(str).fillna('')
        df_keywords['Destaques_Combinados'] = df_keywords[existing_highlight_columns].agg(' '.join, axis=1)
        df_keywords['Destaques_Combinados'] = df_keywords['Destaques_Combinados'].str.strip().str.replace(r'\s+', ' ', regex=True)
        st.write(f"Coluna 'Destaques_Combinados' criada a partir de {', '.join(existing_highlight_columns)}.")

    keywords = ['família', 'internet', 'transporte', 'fome', 'reforço', 'moradia', 'saúde', 'escola', 'violência', 'emprego', 'educação', 'recursos', 'lições de casa']

    keyword_counts = defaultdict(int)
    for text_entry in df_keywords['Destaques_Combinados']:
        text_entry_lower = str(text_entry).lower()
        for keyword in keywords:
            if keyword in text_entry_lower:
                keyword_counts[keyword] += 1

    st.write("Ocorrências de Palavras-Chave nos Destaques Combinados:")
    keyword_counts_str = ""
    if keyword_counts:
        for keyword, count in keyword_counts.items():
            st.write(f"  - {keyword}: {count}")
            keyword_counts_str += f"  - {keyword}: {count}\n"
    else:
        st.write("  Nenhuma das palavras-chave definidas foi encontrada nos destaques.")
        keyword_counts_str = "Nenhuma palavra-chave definida foi encontrada."
    st.markdown("  <small><i>(Nota: A extração de palavras-chave literais pode ser limitada; uma análise de texto mais sofisticada pode ser necessária para insights mais profundos de sentenças.)</i></small>", unsafe_allow_html=True)
    st.markdown("""
    A análise de palavras-chave nos campos 'Destaque IEG' e 'Destaque IDA' revela termos comuns que os alunos mencionam. Essas palavras-chave podem indicar desafios ou necessidades recorrentes (ex: 'família', 'internet', 'fome', 'saúde'), fornecendo pistas qualitativas sobre fatores que impactam o engajamento e o desempenho dos alunos. A frequência de cada palavra-chave destaca as preocupações mais presentes.
    """)
    
    school_inde_treemap = school_inde_stats.reset_index()
    school_inde_treemap.columns = ['Tipo_Escola', 'Media_INDE', 'Desvio_Padrao_INDE', 'Contagem_Alunos']

    school_inde_treemap['Texto_Exibicao'] = school_inde_treemap.apply(lambda row:
        f"Média: {row['Media_INDE']:.2f}<br>Alunos: {row['Contagem_Alunos']}", axis=1)

    fig_treemap = px.treemap(school_inde_treemap,
                     path=['Tipo_Escola'],
                     values='Contagem_Alunos',
                     color='Media_INDE',
                     color_continuous_scale='viridis',
                     title='Média do INDE Consolidado por Tipo de Escola e Contagem de Alunos')

    fig_treemap.update_traces(textfont_size=14)
    st.plotly_chart(fig_treemap)
    st.markdown("""
    O treemap visualiza a distribuição dos alunos por tipo de escola e a média do INDE Consolidado. O tamanho de cada caixa representa o número de alunos, enquanto a cor pode indicar a média do INDE. Isso permite identificar rapidamente onde a 'Passos Mágicos' tem a maior concentração de alunos e como o desempenho médio se distribui entre esses grupos, complementando o gráfico de barras.
    """)

    # --- Geração de Insights com a API do Gemini ---

    prompt_gemini = f"""
    Analise os seguintes dados para gerar sugestões práticas que melhorem a operação da instituição 'Passos Mágicos'.

    **1. Contagem de Alunos por Tipo de Escola:**
    ```
    {ie_counts_str}
    ```

    **2. Estatísticas Agregadas do INDE Consolidado por Tipo de Escola:**
    ```
    {school_inde_stats_str}
    ```

    **3. Ocorrências de Palavras-Chave nos Destaques Combinados (IEG e IDA):**
    ```
    {keyword_counts_str}
    ```

    Com base nesses dados, por favor, forneça:
    1. Quais são as principais diferenças no desempenho (INDE) entre alunos de escolas públicas e privadas? Quais são as implicações para o foco de atuação da 'Passos Mágicos'
    2. Quais palavras-chave nos destaques indicam desafios comuns que os alunos enfrentam e que podem estar impactando o desempenho? Como a 'Passos Mágicos' pode abordar esses desafios?
    3. Sugira ações estratégicas concretas e programas que a instituição pode implementar para otimizar suas operações e melhorar o suporte aos alunos, considerando os insights dos tipos de escola e dos destaques.
    4. Formule a resposta de forma clara e objetiva, adequada para educadores e gestores, utilizando tópicos ou listas para facilitar a leitura.
    """

    if st.button("Gerar Insights Adicionais"):
        gemini_insights = generate_gemini_insights(prompt_gemini)
    
        if gemini_insights:
            st.markdown(gemini_insights)
        else:
            st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


def main():
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Escolha uma Análise", [
        "Visão Geral",
        "Pergunta 1: Adequação do nível (IAN)",
        "Pergunta 2: Desempenho acadêmico (IDA)",
        "Pergunta 3: Engajamento nas atividades (IEG)",
        "Pergunta 4: Autoavaliação (IAA)",
        "Pergunta 5: Aspectos Psicossociais (IPS)",
        "Pergunta 6: Aspectos Psicopedagógicos (IPP)",
        "Pergunta 7: Ponto de Virada (IPV)",
        "Pergunta 8: Multidimensionalidade dos Indicadores",
        "Pergunta 9: Previsão de Risco com Machine Learning",
        "Pergunta 10: Efetividade do Programa",
        "Pergunta 11: Insights e Criatividade"
    ])

    st.title("Análise de Dados de Performance Estudantil")

    if page == "Visão Geral":
        st.subheader("Bem-vindo(a) à Análise de Dados da Passos Mágicos")
        st.write("Utilize o menu lateral para navegar entre as diferentes perguntas e insights gerados. Este dashboard interativo visa explorar as correlações, impactos e padrões nos dados de performance estudantil, autoavaliação, engajamento e aspectos psicossociais e psicopedagógicos.")

        st.markdown("### Visão Geral do Projeto")
        st.markdown("Este projeto teve como objetivo analisar dados de performance estudantil para a **Passos Mágicos**, buscando identificar padrões, correlações e insights acionáveis que possam otimizar suas operações e melhorar o suporte aos alunos.")

        st.markdown("#### Preparação e Limpeza de Dados")
        st.write("O primeiro passo envolveu o carregamento do dataset `PEDE_Completo_Normalizado.csv` e a limpeza rigorosa das colunas numéricas essenciais (IAA, IDA, IEG, IPS, IPV, IPP, IAN, INDE 2022, 2023, 2024). Valores como vírgulas foram substituídos por pontos e os tipos de dados foram convertidos para numéricos, garantindo a consistência para análises subsequentes.")
        st.write("As 5 primeiras linhas do DataFrame após a limpeza inicial das colunas numéricas:")
        st.dataframe(df[numeric_cols_to_clean].head())

        st.markdown("#### Perguntas Chave e Principais Insights")

        st.markdown("**Pergunta 1: Adequação do nível (IAN)**")
        st.write("Apresenta a distribuição do Índice de Desenvolvimento do Aluno (INDE) para o ano de 2024, juntamente com a média e o desvio padrão. Isso fornece uma visão geral do desempenho atual dos alunos.")

        st.markdown("**Pergunta 2: Desempenho acadêmico (IDA)**")
        st.write("Analisa a evolução do INDE médio dos alunos ao longo dos anos (2022-2024) e calcula o Índice de Volatilidade de Aprendizagem (IVA) por Fase, identificando as fases com maior oscilação no desempenho. Insights do Gemini sugerem intervenções para estabilizar o desempenho.")

        st.markdown("**Pergunta 3: Engajamento nas atividades (IEG)**")
        st.write("Examina a relação entre o Índice de Engajamento (IEG) e a probabilidade de um aluno atingir o 'Ponto de Virada' (IPV). Um limiar de IEG é sugerido para identificar alunos em risco, com alertas proativos para intervenções que visam aumentar o engajamento e o potencial de virada. Insights do Gemini oferecem estratégias de intervenção.")

        st.markdown("**Pergunta 4: Autoavaliação (IAA)**")
        st.write("Analisamos a coerência entre a percepção dos alunos sobre si mesmos e seu desempenho/engajamento. Correlações de Pearson e visualizações com linhas de 45 graus ajudam a identificar alunos superestimados, subestimados ou conscientes. Uma 'Matriz de Perfil Psico-Pedagógico' foi criada para categorizar alunos, com insights do Gemini para ações específicas.")

        st.markdown("**Pergunta 5: Aspectos Psicossociais (IPS)**")
        st.write("Investigamos se fatores psicossociais podem prever quedas no Índice de Desenvolvimento do Aluno (INDE). Calculamos deltas do INDE, correlações entre IPS e variações do INDE, e utilizamos testes T e regressões logísticas para identificar padrões. Os insights do Gemini oferecem recomendações para mitigar essas quedas.")

        st.markdown("**Pergunta 6: Aspectos Psicopedagógicos (IPP)**")
        st.write("Comparamos avaliações psicopedagógicas (IPP) com o Índice de Atraso de Desenvolvimento (IAN) para entender se confirmam ou contradizem a defasagem. Coeficientes de Pearson e testes T foram utilizados, com visualizações de dispersão e box plots. O Gemini fornece insights sobre as implicações práticas dessa relação.")

        st.markdown("**Pergunta 7: Ponto de Virada (IPV)**")
        st.write("Utiliza um modelo de Machine Learning (RandomForestClassifier) para prever a probabilidade de um aluno atingir o 'Ponto de Virada'. Inclui sliders interativos para simular o impacto de mudanças em indicadores como IDA, IEG, IPS, IAA e IPP na probabilidade. Insights do Gemini geram um 'Plano de Metas Individualizado' com sugestões práticas.")

        st.markdown("**Pergunta 8: Multidimensionalidade dos Indicadores**")
        st.write("Determinamos quais combinações de indicadores (IDA, IEG, IPS, IPP) mais influenciam a nota global (INDE). Um modelo de Regressão Linear (OLS) foi aplicado, e os alunos foram segmentados em grupos de 'Baixo INDE' e 'Alto INDE' para comparação de perfis através de gráficos de radar. Insights do Gemini sugerem as melhores alavancas para elevar o INDE.")

        st.markdown("**Pergunta 9: Previsão de Risco com Machine Learning**")
        st.write("Apresenta um modelo de Machine Learning (RandomForestClassifier) para prever o risco de defasagem dos alunos, com métricas como AUC e curva ROC. A integração com SHAP permite a explicabilidade do modelo, mostrando os fatores que mais contribuem para o risco de cada aluno. Uma 'Fila de Prioridade' é gerada, e o Gemini oferece recomendações personalizadas para intervenção.")

        st.markdown("**Pergunta 10: Efetividade do Programa**")
        st.write("Calcula o Índice de Valor Adicionado (IVA) para os alunos, medindo o progresso de desempenho em relação à média esperada para sua fase. Visualiza a evolução do INDE por fase e apresenta um 'Selo de Impacto' com o IVA geral da instituição, gerando insights do Gemini para atrair investidores e comprovar a eficácia do método.")

        st.markdown("**Pergunta 11: Insights e Criatividade**")
        st.write("Geramos sugestões práticas adicionais através de cruzamentos de dados. Isso incluiu a consolidação do INDE, análise de desempenho por tipo de instituição de ensino (pública vs. privada) e uma análise de palavras-chave nos destaques dos alunos para identificar desafios comuns. O Gemini compilou recomendações estratégicas baseadas nesses insights.")

        st.markdown("#### Conclusão")
        st.write("Este dashboard serve como uma ferramenta poderosa para a Passos Mágicos, convertendo dados brutos em inteligência acionável para apoiar seus alunos de forma mais eficaz e otimizar suas estratégias educacionais.")

    elif page == "Pergunta 1: Adequação do nível (IAN)":
        display_question_1(df)
    elif page == "Pergunta 2: Desempenho acadêmico (IDA)":
        display_question_2(df)
    elif page == "Pergunta 3: Engajamento nas atividades (IEG)":
        display_question_3(df)
    elif page == "Pergunta 4: Autoavaliação (IAA)":
        display_question_4(df)
    elif page == "Pergunta 5: Aspectos Psicossociais (IPS)":
        display_question_5(df)
    elif page == "Pergunta 6: Aspectos Psicopedagógicos (IPP)":
        display_question_6(df)
    elif page == "Pergunta 7: Ponto de Virada (IPV)":
        display_question_7(df)
    elif page == "Pergunta 8: Multidimensionalidade dos Indicadores":
        display_question_8(df)
    elif page == "Pergunta 9: Previsão de Risco com Machine Learning":
        display_question_9(df)
    elif page == "Pergunta 10: Efetividade do Programa":
        display_question_10(df)
    elif page == "Pergunta 11: Insights e Criatividade":
        display_question_11(df, numeric_cols_to_clean)

if __name__ == '__main__':
    main()









