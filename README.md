# Restoring ancient text using deep learning
## A case study on Greek epigraphy

Yannis Assael<sup>\*</sup>, Thea Sommerschield<sup>\*</sup>, Jonathan Prag

---

Ancient History relies on disciplines such as Epigraphy, the study of ancient inscribed texts, for evidence of the recorded past. However, these texts, "inscriptions", are often damaged over the centuries, and illegible parts of the text must be restored by specialists, known as epigraphists.
This work presents a novel assistive method for providing text restorations using deep neural networks.
To the best of our knowledge, Pythia is the first ancient text restoration model that recovers missing characters from a damaged text input.
Its architecture is carefully designed to handle long-term context information, and deal efficiently with missing or corrupted character and word representations. 
To train it, we wrote a non-trivial pipeline to convert PHI, the largest digital corpus of ancient Greek inscriptions, to machine actionable text, which we call PHI-ML.
On PHI-ML, Pythia's predictions achieve a 30.1% character error rate, compared to the 57.3% of human epigraphists. Moreover, in 73.5% of cases the ground-truth sequence was among the Top-20 hypotheses of Pythia, which effectively demonstrates the impact of such an assistive method on the field of digital epigraphy, and sets the state-of-the-art in ancient text restoration.


<p align="center">
<img alt="Pythia architecture" src="http://yannisassael.com/projects/pythia/pythia_arch.png" width="75%" /><br />
Pythia-Bi-Word processing the phrase μηδέν ἄγαν (mēdén ágan) "nothing in excess", a fabled maxim inscribed on Apollo's temple in Delphi. The letters "γα" are missing, and annotated with "?". Since word ἄ??ν contains missing characters, its embedding is treated as unknown ("unk"). The decoder outputs correctly "γα".
</p>


### References

- [arXiv pre-print](https://arxiv.org/abs/1910.06262)
- [EMNLP-IJCNLP 2019](https://www.emnlp-ijcnlp2019.org/program/accepted/)

When using any of this project's source code, please cite:
```
@inproceedings{assael2019restoring,
  title={Restoring ancient text using deep learning: a case study on {Greek} epigraphy},
  author={Assael, Yannis and Sommerschield, Thea and Prag, Jonathan},
  booktitle={Empirical Methods in Natural Language Processing},
  year={2019}
}
```

## Pythia online

To aid further research in the field we created an online interactive python notebook, where researchers can query one of our models to get text restorations and visualise the attention weights.

- [Google Colab](https://colab.research.google.com/drive/16RfCpZLm0M6bf3eGIA7VUPclFdW8P8pZ)

## Pythia offline

The following snippets provide references for regenerating PHI-ML and training new models offline.

#### Dependencies
```
pip install requests bs4 coloredlogs dm-sonnet editdistance lxml nltk tensor2tensor tensorflow-gpu tqdm && \
python -m nltk.downloader punkt
```

#### PHI-ML dataset generation
```
# Download PHI (this will take a while)
python -c 'import pythia.data.phi_download; pythia.data.phi_download.main()'

# Process and generate PHI-ML
python -c 'import pythia.data.phi_process; pythia.data.phi_process.main()'
```

#### Training
```
python -c 'import pythia.train; pythia.train.main()'
```

#### Evaluation
```
python -c 'import pythia.test; pythia.test.main()' --load_checkpoint="your_model_path/"
```

#### Docker execution
```
./build.sh
./run.sh <GPU_ID> python -c 'import pythia.train; pythia.train.main()'
```

## License
Apache License, Version 2.0

<p align="center">
<img alt="Epigraphy" src="http://yannisassael.com/projects/pythia/epigraphy_transp.png" width="256" /><br />
Damaged inscription: a decree concerning the Acropolis of Athens (485/4 BCE). <it>IG</it> I<sup>3</sup> 4B.<br />(CC BY-SA 3.0, WikiMedia)
</p>
