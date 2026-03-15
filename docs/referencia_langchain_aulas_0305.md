# Referência – Aulas LangChain (PDFs 030501–030506)

Resumo dos 6 PDFs de `FASE3/pdfs/` para alinhar o **Step 5 (Assistente com LangChain)** ao que foi apresentado.

---

## 030501 – Introdução a LangChain

- **LLM:** `from langchain_openai import OpenAI`; `load_dotenv()`; `llm = OpenAI(api_key=...)`; `response = llm.invoke(prompt)`.
- **Output parsers:** estruturar saída (ex.: JSON) com `PydanticOutputParser`.
- **Memória:** `ConversationBufferMemory()`; `memory.add_messages(...)`; usar `memory.buffer` no prompt.
- **Chains:** encadear etapas – saída de uma vira entrada da próxima (`primeira_etapa(input) → segunda_etapa(analise)`).
- **Componentes:** Document Loaders, Prompts, Chains, LLM, Agents.

---

## 030502 – Document Loaders

- **PyPDFLoader:** `PyPDFLoader(caminho_pdf)` → `loader.load()` → lista de documentos com `.page_content`.
- **CSVLoader:** `CSVLoader(caminho_csv)` → cada linha vira um documento.
- **WebBaseLoader:** `WebBaseLoader(url)` → conteúdo de página web.
- **Uso no assistente:** carregar contexto opcional (ex.: PDF, texto) para enriquecer a pergunta antes de enviar à LLM.

---

## 030503 – Prompts

- **PromptTemplate:** `PromptTemplate(template=..., input_variables=["var1", "var2"])`; `prompt.format(var1=..., var2=...)`; `llm.invoke(prompt_completo)`.
- **Estrutura do prompt:** Instrução, Contexto, Dados de entrada, Indicador de saída.
- **Exemplos:** geração de texto, extração de informações, sumarização; clareza e especificidade.

---

## 030504 – Chains

- **LLMChain:** `LLMChain(llm=llm, prompt=prompt_template)`; `chain.invoke({"input_var": value})` com `output_key` opcional.
- **SequentialChain:** `SequentialChain(memory=SimpleMemory(memories={...}), chains=[c1, c2, c3], input_variables=[...], output_variables=[...])`; `cadeia_sequencial.run(input=...)`.
- **Router Chain:** rotear por conteúdo (ex.: tipo de pergunta) e escolher qual chain executar; LCEL: `prompt | llm | StrOutputParser()` e `with_structured_output` para decisão de rota.

---

## 030505 – Integração com LLM

- **Ollama (local):** `from langchain_community.llms import Ollama`; `Ollama(model="llama2", ...)`; trocar modelo (ex.: mistral) alterando só o nome.
- **Fluxo:** carregar documentos (TextLoader) → para cada doc → `chain_sentimento.invoke({"text": ...})`, `chain_resumo.invoke({"text": ...})`.
- **No nosso caso:** usar LLM local (Qwen + PEFT) via `HuggingFacePipeline` ou equivalente no LangChain.

---

## 030506 – Criação de Agents

- **Agent:** perceber ambiente → decidir → agir.
- **Exemplo:** pergunta do usuário → consultar DB (ex.: PostgreSQL) → montar prompt com dados → `llm(prompt)` → resposta.
- **Padrão:** `responder_pergunta(pergunta)` → `dados = obter_dados_*(pergunta)` → se houver dados, montar contexto e chamar LLM; senão, resposta padrão. Incluir “fonte” (ex.: “Insights da OpenAI” / dados do DB).

---

## Aplicação ao Step 5 (Assistente médico)

1. **Carregar LLM fine-tunada:** módulo que carrega Qwen+PEFT e expõe como LLM do LangChain (ex.: pipeline HuggingFace → `HuggingFacePipeline`).
2. **Prompts:** `PromptTemplate` com variáveis como `{contexto}` (opcional) e `{pergunta}`; instrução de papel (assistente médico), indicador de saída e disclaimer.
3. **Chain:** (opcional) obter contexto (Document Loader ou “consulta DB”) → montar prompt (contexto + pergunta) → LLM → resposta.
4. **Segurança e explainability:** texto fixo de disclaimer (“não substitui médico”); citar fonte quando houver contexto (ex.: “Com base no contexto fornecido…”).

Referência: `FASE3/pdfs/030501` a `030506`.
