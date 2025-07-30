


## Análise de Sinais de Temperatura usando FFT

Este projeto realiza o tratamento e análise de sinais de temperatura utilizando técnicas de Transformada Rápida de Fourier (FFT), filtragem, suavização e regressão linear.

---

## 📌 Objetivo do Projeto

Desenvolver ferramentas em Python para:

- Ler e processar sinais de temperatura a partir de arquivos CSV.
- Aplicar filtros e transformações (FFT, média móvel).
- Detectar máximos, mínimos e realizar regressão linear.
- Visualizar resultados gráficos para interpretação.

---

## 🗂️ Organização do Projeto

```
onset_calculation/
│
├── data/           # Arquivos CSV de entrada
├── results/        # Resultados gerados (ex: media_movel.csv)
├── src/            # Scripts Python
├── README.md
└── requirements.txt
```

**Recomendações:**
- Guarde todos os arquivos CSV de entrada na pasta `data/`.
- Coloque os scripts em `src/` e use nomes descritivos como `signal_analysis.py`.
- Os resultados gerados (arquivos, gráficos) vão em `results/`.

---

## 🛠️ Como executar

1. (Opcional) Criar um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Instalar as dependências:
   ```bash
   pip install numpy scipy matplotlib pandas scikit-learn sympy
   ```

3. Executar o script principal:
   ```bash
   src/PyCode_FFT_Signal_Treatment3.py
   ```

---

## 📄 Descrição dos Arquivos Importantes

- `data/`: Arquivos de dados de entrada (CSV).
- `src/`: Scripts de análise e plotagem.
- `results/`: Arquivos gerados (ex: media_movel.csv).
- `README.md`: Documentação do projeto.
- `requirements.txt`: Lista de dependências (pode ser gerada com `pip freeze > requirements.txt`).

---

## 📝 Convenções de Nomes

- Use nomes em inglês e descritivos para scripts: `signal_analysis.py`, `fft_filter.py`, etc.
- Os dados originais devem ir em `data/` e os resultados em `results/`.
- Evite espaços e caracteres especiais nos nomes dos arquivos.

---

## 🔜 Próximos passos

- Refatorar o código e separar funções em módulos se necessário.
- Melhorar a documentação e adicionar exemplos de uso.
- Automatizar a geração de gráficos e resultados.

---

## 👩‍💻 Autoria

Projeto desenvolvido no âmbito da Iniciação Científica (IC) da Natalia.

