- corpus_files_gen.py

to generate both intial train corpus (default: 50000 examples, each 20000 bp with 200 bp overlap) and whole train corpus to train tokenizer

- token_gen.py

Use intial train corpus to train a initial base tokenizer, then call its fast version to train on the whole train corpus.


multiple specice: (20000-200)
DatasetDict({
    train: Dataset({
        features: ['sequence', 'start_pos', 'end_pos', 'description', 'species', 'fna_url'],
        num_rows: 8548111
    })
    test: Dataset({
        features: ['sequence', 'start_pos', 'end_pos', 'description', 'species', 'fna_url'],
        num_rows: 9891
    })
})


>>> working on chm13_t2t_20000_200
DatasetDict({
    train: Dataset({
        features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
        num_rows: 151679
    })
    test: Dataset({
        features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
        num_rows: 2592
    })
})
>>> working on crgd_t2t_20000_200
DatasetDict({
    train: Dataset({
        features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
        num_rows: 153725
    })
    test: Dataset({
        features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
        num_rows: 2539
    })
})

NOTE we use 400000 sentence to train token for multi specice.


