# 🛡️ Modelo de Detecção de Textos Maliciosos com PLN e Aprendizado de Máquina
> **Benchmark de Representação Vetorial:** Bag of Words (BoW) vs. TF-IDF

Este projeto consiste na construção e avaliação de modelos de **Processamento de Linguagem Natural (PLN)** e **Aprendizado de Máquina (Machine Learning)** para classificar automaticamente mensagens de texto entre conteúdos benignos e diferentes modalidades de ciberameaças.

---

## 📊 1. Base de Dados
O projeto utiliza o conjunto de dados **`TExtPhish`** (disponível no HuggingFace), contendo **123.438 amostras** extraídas do Reddit e da base de e-mails da Enron.

O modelo executa uma **classificação multiclasse** cobrindo 4 categorias:
* `benign`: Mensagens legítimas/benignas (*83.106 amostras*)
* `sextortion`: Chantagens com conteúdo de sextorsão (*24.724 amostras*)
* `blackmail`: Mensagens de chantagem e extorsão geral (*8.880 amostras*)
* `ransomware`: Ameaças relacionadas a sequestro de dados (*6.728 amostras*)

---

## 🛠️ 2. Pipeline de Processamento (PLN)

Para garantir a limpeza dos dados e extração eficiente de características textuais, foi aplicado o seguinte pipeline utilizando a biblioteca **NLTK**:

1. **Normalização:** Conversão de todo o texto para minúsculas (`lowercase`) e remoção de espaços nas extremidades.
2. **Tokenização e Limpeza:** Divisão do texto em tokens mantendo apenas caracteres alfanuméricos (`isalnum`), eliminando pontuações e símbolos.
3. **Remoção de Stopwords:** Filtragem de palavras frequentes em inglês sem valor semântico relevante.
4. **Lematização com POS-Tagging:** Redução das palavras às suas formas gramaticais fundamentais via `WordNetLemmatizer` acoplado ao `pos_tag`.

---

## 📐 3. Engenharia de Recursos e Vetorização

O dataset foi dividido estrategicamente em **80% para treino** e **20% para teste** de forma estratificada. Foram comparadas duas abordagens tradicionais de vetorização:

* **Bag of Words (CountVectorizer):** Mapeamento de frequência absoluta com vocabulário expandido para **170.079 recursos**.
* **TF-IDF (TfidfVectorizer):** Mapeamento com Unigramas e Bigramas `(1, 2)` limitados às **5.000 características** mais relevantes.
* **Padronização:** Aplicação de `StandardScaler(with_mean=False)` preservando a esparsidade das matrizes de texto.

---

## 🤖 4. Modelagem e Hiperparâmetros

A otimização dos hiperparâmetros foi realizada através do **`GridSearchCV`** com validação cruzada (3 *folds*), testando os seguintes algoritmos:

* **Regressão Logística** (Ajuste do parâmetro de regularização $C$)
* **K-Nearest Neighbors - KNN** (Variação do número de vizinhos $k$)
* **Redes Neurais / MLPClassifier** (Múltiplas arquiteturas com 3 camadas ocultas variando de 2 a 32 neurônios)
* **Naïve Bayes (GaussianNB)**

---

## 🏆 5. Resultados e Benchmark

Resultados obtidos durante a avaliação final no conjunto de **teste** (amostras inéditas):

| Posição | Experimento / Modelo | Acurácia de Teste | Melhores Hiperparâmetros |
| :---: | :--- | :---: | :--- |
| 🥇 **1º** | **BoW + Regressão Logística** | **84,39%** | `C: 0.001` |
| 🥈 **2º** | **TF-IDF + Regressão Logística** | **83,11%** | `C: 0.001` |
| 🥉 **3º** | **TF-IDF + Rede Neural** | **79,67%** | `hidden_layer_sizes: (32, 32, 32)` |
| 4º | **TF-IDF + KNN** | **78,01%** | `n_neighbors: 12` |
| 5º | **BoW + Rede Neural** | **77,98%** | `hidden_layer_sizes: (32, 32, 32)` |
| 6º | **BoW + KNN** | **74,83%** | `n_neighbors: 4` |

### 🎯 Métricas do Modelo Campeão (`BoW + Regressão Logística`):
* **Acurácia:** 84,39%
* **Precision (Weighted):** 83,63%
* **Recall (Weighted):** 84,39%

---

## 💡 6. Conclusões
* A combinação **Bag of Words + Regressão Logística** apresentou o melhor equilíbrio entre tempo de processamento e desempenho de classificação.
* Modelos lineares demonstraram eficácia superior e menor consumo computacional quando comparados a modelos baseados em distância (KNN) ou redes neurais multicamadas simples (MLP) para esta tarefa.
