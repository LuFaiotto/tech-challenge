!pip install -q streamlit pandas numpy seaborn matplotlib scipy statsmodels scikit-learn plotly google-generativeai

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

# Gemini API imports and configuration
import google.generativeai as genai
from google.colab import userdata # Only for Colab environment setup

# Definindo as colunas numéricas que precisam de tratamento globalmente
numeric_cols_to_clean = [
    "IAA", "IDA", "IEG", "IPS", "IPV", "IPP", "IAN",
    "INDE 2022", "INDE 2023", "INDE 2024"
]

# Função para converter e limpar colunas numéricas
def clean_numeric_column(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

# Lendo arquivo do Github para criação de dataframe
df = pd.read_csv('/PEDE_Completo_Normalizado.csv', sep=';', engine='python', encoding='latin1')
pd.options.display.float_format = '{:,.2f}'.format

# Aplicar a função de limpeza a todas as colunas identificadas
for col in numeric_cols_to_clean:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])

# Exemplo de uso ou verificação inicial (opcional para o app.py, mas útil para demonstração)
# st.write("Limpeza de dados para colunas numéricas concluída.")
# st.dataframe(df[numeric_cols_to_clean].head())

# --- Configuração da API do Gemini ---
# A chave da API do Gemini deve ser configurada ANTES de chamar generate_gemini_insights
# Em um ambiente Streamlit local ou em nuvem, você usaria st.secrets["GEMINI_API_KEY"]
# Para este ambiente Colab, vamos simular isso para que a função possa ser testada
try:
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY') # Tenta obter do Colab Secrets
    genai.configure(api_key=GEMINI_API_KEY)
except userdata.SecretNotFoundError:
    GEMINI_API_KEY = None # Se não encontrar, define como None
    st.warning("ATENÇÃO: A chave 'GEMINI_API_KEY' não foi encontrada nos Secrets do Colab.")
    st.warning("Por favor, adicione sua chave da API do Gemini como um Secret do Colab para usar os insights do Gemini.")
    st.warning("Vá em 'Secrets' (ícone de chave no painel esquerdo) e adicione 'GEMINI_API_KEY' com sua chave.")
except Exception as e:
    GEMINI_API_KEY = None
    st.error(f"Erro ao configurar a API do Gemini: {e}")

# Função genérica para gerar insights com Gemini
def generate_gemini_insights(prompt):
    if GEMINI_API_KEY is None:
        return "Erro: A chave da API do Gemini não está configurada. Por favor, configure-a nos Secrets do Colab."
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        # Verifica se há conteúdo na resposta antes de tentar acessá-lo
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return f"A API do Gemini não retornou conteúdo. Possíveis bloqueios: {response.prompt_feedback}"
    except Exception as e:
        return f"Erro ao gerar insights com Gemini: {e}"


# --- Otimização da Pergunta 4 (Ponto 2 do plano) ---
# Função para gerar gráficos de regressão com linha de 45 graus
def plot_reg_with_45_deg_line(data, x_col, y_col, title_pt, x_label_pt, y_label_pt):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_col, y=y_col, data=data, scatter_kws={"s": 50, "alpha": 0.7},
                line_kws={"color": "red", "label": "Linha de Regressão"})
    plt.title(title_pt)
    plt.xlabel(x_label_pt)
    plt.ylabel(y_label_pt)
    # Define os limites para a linha de 45 graus com base nos limites atuais dos eixos
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, color='purple', linestyle='--', linewidth=1, label='Linha de 45°')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

# Modificação da função display_question_4 para incluir a Matriz de Perfil Psico-Pedagógico e insights do Gemini
def display_question_4(df):
    st.header("Pergunta 4: Autoavaliação (IAA) vs Desempenho Real (IDA) e Engajamento (IEG)")
    st.markdown("***As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?***")

    # Criando um sub-dataframe limpo para esta análise específica
    df_clean_q4 = df.dropna(subset=["IAA", "IDA", "IEG"]).copy()

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

    st.subheader("Visualizações")

    # Gráfico 1: IAA (autoavaliação) vs IDA (desempenho real)
    plot_reg_with_45_deg_line(df_clean_q4, "IDA", "IAA",
                              "Autoavaliação (IAA) vs Desempenho Real (IDA)",
                              "IDA (Desempenho Real)", "IAA (Autoavaliação)")
    st.pyplot(plt.gcf())
    plt.close() # Fecha a figura para evitar sobreposição em futuras chamadas

    # Gráfico 2: IAA (autoavaliação) vs IEG (engajamento)
    plot_reg_with_45_deg_line(df_clean_q4, "IEG", "IAA",
                              "Autoavaliação (IAA) vs Engajamento (IEG)",
                              "IEG (Engajamento)", "IAA (Autoavaliação)")
    st.pyplot(plt.gcf())
    plt.close() # Fecha a figura

    # Gráfico 3: IAA vs IDA por Sexo
    plt.figure(figsize=(10, 7))
    # sns.lmplot retorna um FacetGrid, que contém a figura. Precisamos acessar a figura.
    g = sns.lmplot(x="IDA", y="IAA", hue="Sexo", data=df_clean_q4, height=6, aspect=1.2,
               markers=["o", "s"], scatter_kws={"s": 60, "alpha": 0.7})
    g.fig.suptitle("Autoavaliação (IAA) vs Desempenho Real (IDA) por Sexo", y=1.02) # Ajusta o título para FacetGrid
    g.set_axis_labels("IDA (Desempenho Real)", "IAA (Autoavaliação)")
    g.add_legend(title="Sexo")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(g.fig)
    plt.close(g.fig) # Fecha a figura do FacetGrid

    # --- Nova Análise: Matriz de Perfil Psico-Pedagógico (IDA e IAA) ---
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

    # Gráfico de barras para a distribuição dos quadrantes
    fig_quadrant, ax_quadrant = plt.subplots(figsize=(10, 6))
    sns.barplot(x=quadrant_analysis_df.index, y=quadrant_analysis_df['Percentual (%)'], palette='viridis', ax=ax_quadrant)
    ax_quadrant.set_title('Percentual de Alunos por Perfil Psico-Pedagógico')
    ax_quadrant.set_xlabel('Perfil')
    ax_quadrant.set_ylabel('Percentual de Alunos (%)')
    ax_quadrant.tick_params(axis='x', rotation=45)
    ax_quadrant.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_quadrant)
    plt.close(fig_quadrant)

    # --- Geração de Insights com a API do Gemini ---
    st.subheader("Insights do Gemini sobre a Matriz de Perfil Psico-Pedagógico")

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

    gemini_insights = generate_gemini_insights(prompt_gemini)

    if gemini_insights:
        st.markdown(gemini_insights)
    else:
        st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")

# Modified analyze_and_plot_queda_streamlit to return stats
def analyze_and_plot_queda_streamlit(data, year_pair_str, ips_col='IPS', other_cols=['Idade', 'Sexo', 'IEG', 'IDA']):
    queda_col = f'queda_{year_pair_str}'
    delta_inde_col = f'delta_INDE_{year_pair_str}'

    st.subheader(f"Análise para a Queda {year_pair_str}:")

    results_for_prompt = {}

    # Média de IPS para grupos com e sem queda
    st.markdown(f"**IPS médio (queda {year_pair_str} vs. sem queda):**")
    ips_mean_stats = data.groupby(queda_col)[ips_col].agg(['count', 'mean', 'std'])
    st.dataframe(ips_mean_stats)
    results_for_prompt['ips_mean_stats'] = ips_mean_stats.to_string()

    # T-test
    grp1 = data.loc[data[queda_col] == True, ips_col].dropna()
    grp0 = data.loc[data[queda_col] == False, ips_col].dropna()
    if len(grp1) > 1 and len(grp0) > 1:
        tstat, pval = ttest_ind(grp1, grp0, grp0, equal_var=False)
        st.write(f"**Teste T de IPS (queda vs. sem queda):** t = {tstat:.3f}, p = {pval:.4f}")
        results_for_prompt['ttest_results'] = f"t = {tstat:.3f}, p = {pval:.4f}"
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
            st.text(logit.summary().as_text())
            results_for_prompt['logit_summary'] = logit.summary().as_text()
        except Exception as e:
            st.write(f"Erro ao executar a Regressão Logística para {year_pair_str}: {e}")
            results_for_prompt['logit_summary'] = f"Erro ao executar a Regressão Logística: {e}"
    else:
        st.write(f"Não há dados suficientes para a Regressão Logística para {year_pair_str}.")
        results_for_prompt['logit_summary'] = f"Não há dados suficientes para a Regressão Logística para {year_pair_str}."

    # Plotagem - Scatter Plot
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

    # Plotagem - Box Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x=queda_col, y=ips_col, hue=queda_col, palette='pastel', legend=False, ax=ax2)
    ax2.set_title(f'Distribuição de IPS por Ocorrência de Queda ({year_pair_str})')
    ax2.set_xlabel(f'Queda {year_pair_str} (Verdadeiro/Falso)')
    ax2.set_ylabel('IPS (Aspectos Psicossociais)')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)

    return results_for_prompt


def display_question_5(df):
    st.header("Pergunta 5: Aspectos Psicossociais (IPS) e Quedas de Desempenho")
    st.markdown("***Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?***")

    # Calcular deltas para INDE
    df['delta_INDE_22_23'] = df['INDE 2023'] - df['INDE 2022']
    df['delta_INDE_23_24'] = df['INDE 2024'] - df['INDE 2023']

    threshold = -0.3  # Um delta negativo indica redução, usamos -0.3 como um limiar para 'queda'
    df['queda_22_23'] = df['delta_INDE_22_23'] <= threshold
    df['queda_23_24'] = df['delta_INDE_23_24'] <= threshold

    # Correlação entre IPS e as variações de INDE
    corr_ips_delta = df[['IPS', 'delta_INDE_22_23', 'delta_INDE_23_24']].corr()
    st.subheader("Correlação (IPS vs variações no INDE):")
    st.dataframe(corr_ips_delta['IPS']) # Changed to st.dataframe for better display

    # Store correlation results for prompt
    corr_ips_delta_str = corr_ips_delta['IPS'].to_string()

    # Chamar a função para os dois períodos de queda e capturar os resultados
    results_22_23 = analyze_and_plot_queda_streamlit(df, '22_23')
    results_23_24 = analyze_and_plot_queda_streamlit(df, '23_24')

    # --- Geração de Insights com a API do Gemini ---
    st.subheader("Insights do Gemini sobre Aspectos Psicossociais e Quedas de Desempenho")

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

    gemini_insights = generate_gemini_insights(prompt_gemini)

    if gemini_insights:
        st.markdown(gemini_insights)
    else:
        st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


def display_question_6(df):
    st.header("Pergunta 6: Aspectos Psicopedagógicos (IPP) e Defasagem (IAN)")
    st.markdown("***As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?***")

    df_clean_q6 = df.dropna(subset=["IPP", "IAN"]).copy()

    # Correlação de Pearson entre IPP e IAN
    correlation_ipp_ian, p_value_ipp_ian = pearsonr(df_clean_q6['IPP'], df_clean_q6['IAN'])

    st.subheader("Análise de Correlação")
    st.write(f"- **Correlação de Pearson entre IPP e IAN**: {correlation_ipp_ian:.3f}")
    st.write(f"- **Valor-P**: {p_value_ipp_ian:.3f}")

    # Mediana do IAN para categorização
    median_ian = df_clean_q6['IAN'].median()
    st.write(f"- **Mediana do IAN**: {median_ian:.2f}")

    df_clean_q6['IAN_category'] = df_clean_q6['IAN'].apply(lambda x: 'Baixo IAN' if x <= median_ian else 'Alto IAN')
    st.subheader("Categorização do IAN")
    st.dataframe(df_clean_q6[['IAN', 'IAN_category']].head())

    # Teste T de amostras independentes para comparar o IPP entre as categorias de IAN
    ipp_low_ian = df_clean_q6.loc[df_clean_q6['IAN_category'] == 'Baixo IAN', 'IPP']
    ipp_high_ian = df_clean_q6.loc[df_clean_q6['IAN_category'] == 'Alto IAN', 'IPP']

    t_statistic, p_value_ttest, mean_ipp_low_ian, mean_ipp_high_ian = None, None, None, None

    if len(ipp_low_ian) > 1 and len(ipp_high_ian) > 1:
        t_statistic, p_value_ttest = ttest_ind(ipp_low_ian, ipp_high_ian, equal_var=False)
        st.subheader("Teste T comparando IPP entre 'Baixo IAN' e 'Alto IAN':")
        st.write(f"- **Estatística T**: {t_statistic:.3f}")
        st.write(f"- **Valor-P**: {p_value_ttest:.3f}")

        # Média de IPP para ambos os grupos
        mean_ipp_low_ian = ipp_low_ian.mean()
        mean_ipp_high_ian = ipp_high_ian.mean()

        st.write(f"- **IPP Médio para o grupo 'Baixo IAN'**: {mean_ipp_low_ian:.3f}")
        st.write(f"- **IPP Médio para o grupo 'Alto IAN'**: {mean_ipp_high_ian:.3f}")
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

    # Box Plot de IPP por Categoria de IAN
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_clean_q6, x='IAN_category', y='IPP', hue='IAN_category', palette='pastel', legend=False, ax=ax2)
    ax2.set_title('Box Plot de IPP por Categoria de IAN')
    ax2.set_xlabel('Categoria de IAN')
    ax2.set_ylabel('IPP (Índice de Perfil Psicopedagógico)')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    plt.close(fig2)

    # --- Geração de Insights com a API do Gemini ---
    st.subheader("Insights do Gemini sobre IPP e IAN")

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

    gemini_insights = generate_gemini_insights(prompt_gemini)

    if gemini_insights:
        st.markdown(gemini_insights)
    else:
        st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


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
    st.text(regression_model.summary().as_text())
    ols_summary_text = regression_model.summary().as_text() # Capture for prompt

    # Identificação de alunos com baixo e alto INDE
    q1_inde_2023 = df_regression_q8['INDE 2023'].quantile(0.25)
    q3_inde_2023 = df_regression_q8['INDE 2023'].quantile(0.75)

    st.subheader("Análise de Percentis do INDE 2023")
    st.write(f"- **25º percentil (Q1) do INDE 2023**: {q1_inde_2023:.2f}")
    st.write(f"- **75º percentil (Q3) do INDE 2023**: {q3_inde_2023:.2f}")

    low_inde_students = df_regression_q8[df_regression_q8['INDE 2023'] <= q1_inde_2023].copy()
    high_inde_students = df_regression_q8[df_regression_q8['INDE 2023'] >= q3_inde_2023].copy()

    st.write("5 primeiras linhas de alunos com 'Baixo INDE':")
    st.dataframe(low_inde_students[['INDE 2023']].head())
    st.write("5 primeiras linhas de alunos com 'Alto INDE':")
    st.dataframe(high_inde_students[['INDE 2023']].head())

    # Normalização e cálculo de pontuações médias
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
    comparison_df_str = comparison_df.to_string() # Capture for prompt

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

    # --- Geração de Insights com a API do Gemini ---
    st.subheader("Insights do Gemini sobre a Multidimensionalidade dos Indicadores")

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

    gemini_insights = generate_gemini_insights(prompt_gemini)

    if gemini_insights:
        st.markdown(gemini_insights)
    else:
        st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


def display_question_11(df, numeric_cols_to_clean):
    st.header("Pergunta 11: Insights Adicionais e Criatividade")
    st.markdown("***Utilize cruzamentos de dados não solicitados nas perguntas anteriores para gerar sugestões práticas que melhorem a operação da instituição.***")

    # Consolidação do INDE
    df['INDE_Consolidado'] = df[numeric_cols_to_clean[-3:]].mean(axis=1, skipna=True)

    st.subheader("Análise do INDE Consolidado por Tipo de Escola")
    st.write("5 primeiras linhas com 'INDE 2022', 'INDE 2023', 'INDE 2024' e 'INDE_Consolidado':")
    st.dataframe(df[numeric_cols_to_clean[-3:] + ['INDE_Consolidado']].head())

    # Limpeza e padronização da coluna 'IE' (Instituição de Ensino)
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
    ie_counts_str = df_clean_school['IE_Limpo'].value_counts().to_string() # Capture for prompt

    # Estatísticas do INDE consolidado por tipo de escola
    school_inde_stats = df_clean_school.groupby('IE_Limpo')['INDE_Consolidado'].agg(['mean', 'std', 'count'])

    st.subheader("Estatísticas Agregadas do INDE Consolidado por Tipo de Escola:")
    st.dataframe(school_inde_stats)
    school_inde_stats_str = school_inde_stats.to_string() # Capture for prompt

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


    # Treemap: Média do INDE Consolidado por Tipo de Escola e Contagem de Alunos
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

    # --- Geração de Insights com a API do Gemini ---
    st.subheader("Insights do Gemini para Sugestões Práticas")

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

    gemini_insights = generate_gemini_insights(prompt_gemini)

    if gemini_insights:
        st.markdown(gemini_insights)
    else:
        st.warning("Não foi possível gerar insights do Gemini. Verifique a configuração da API.")


def main():
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Escolha uma Análise", [
        "Visão Geral",
        "Pergunta 4",
        "Pergunta 5",
        "Pergunta 6",
        "Pergunta 8",
        "Pergunta 11"
    ])

    st.title("Análise de Dados de Performance Estudantil")

    if page == "Visão Geral":
        st.subheader("Bem-vindo(a) à Análise de Dados da Passos Mágicos")
        st.write("Utilize o menu lateral para navegar entre as diferentes perguntas e insights gerados.")
        st.write("Este dashboard interativo permite explorar as correlações, impactos e padrões nos dados de performance estudantil, autoavaliação, engajamento e aspectos psicossociais e psicopedagógicos.")
        st.subheader("Dados Carregados e Limpos:")
        st.write("As 5 primeiras linhas do DataFrame após a limpeza inicial das colunas numéricas:")
        st.dataframe(df[numeric_cols_to_clean].head())

    elif page == "Pergunta 4":
        display_question_4(df)
    elif page == "Pergunta 5":
        display_question_5(df)
    elif page == "Pergunta 6":
        display_question_6(df)
    elif page == "Pergunta 8":
        display_question_8(df)
    elif page == "Pergunta 11":
        display_question_11(df, numeric_cols_to_clean)

if __name__ == '__main__':
    main()