const tf = require('@tensorflow/tfjs-core');
const fs = require('fs');

const { stringToChars } = require('../util');

const { Trie } = require('./trie');

const separator =
    '\u2581';  // This is the unicode character 'lower one eighth block'.

function processInput(str) {
    const normalized = str.normalize('NFKC');
    return normalized.length > 0 ?
        separator + normalized.replace(/ /g, separator) :
        normalized;
}

// The first tokens are reserved for unk, control symbols, and user-defined
// symbols.
const RESERVED_SYMBOLS_COUNT = 6;

class Tokenizer {
    constructor(vocabulary, reservedSymbolsCount = RESERVED_SYMBOLS_COUNT) {
        this.trie = new Trie();
        this.vocabulary = vocabulary;
        this.reservedSymbolsCount = reservedSymbolsCount;

        for (let i = this.reservedSymbolsCount; i < this.vocabulary.length; i++) {
            this.trie.insert(this.vocabulary[i][0], this.vocabulary[i][1], i);
        }
    }

    encode(input) {
        const nodes = [];
        const words = [];
        const best = [];

        input = processInput(input);

        const symbols = stringToChars(input);

        for (let i = 0; i <= symbols.length; i++) {
            nodes.push({});
            words.push(0);
            best.push(0);
        }

        // Construct the lattice.
        for (let i = 0; i < symbols.length; i++) {
            const matches = this.trie.commonPrefixSearch(symbols.slice(i));

            for (let j = 0; j < matches.length; j++) {
                const piece = matches[j];
                const obj = { key: piece[0], score: piece[1], index: piece[2] };

                const endPos = piece[0].length;
                if (nodes[i + endPos][i] == null) {
                    nodes[i + endPos][i] = [];
                }

                nodes[i + endPos][i].push(obj);
            }
        }

        for (let endPos = 0; endPos <= symbols.length; endPos++) {
            for (const startPos in nodes[endPos]) {
                const arr = nodes[endPos][startPos];

                for (let j = 0; j < arr.length; j++) {
                    const word = arr[j];
                    const score = word.score + best[endPos - word.key.length];

                    if (best[endPos] === 0 || score >= best[endPos]) {
                        best[endPos] = score;
                        words[endPos] = arr[j].index;
                    }
                }
            }
        }

        const results = [];

        // Backward pass.
        let iter = words.length - 1;
        while (iter > 0) {
            results.push(words[iter]);
            iter -= this.vocabulary[words[iter]][0].length;
        }

        // Merge consecutive unks.
        const merged = [];
        let isPreviousUnk = false;
        for (let i = 0; i < results.length; i++) {
            const id = results[i];
            if (!(isPreviousUnk && id === 0)) {
                merged.push(id);
            }

            isPreviousUnk = id === 0;
        }

        return merged.reverse();
    }
}

/**
 * Load the Tokenizer for use independently from the UniversalSentenceEncoder.
 *
 * @param pathToVocabulary (optional) Provide a path to the vocabulary file.
 */
async function loadTokenizer(pathToVocabulary) {
    const vocabulary = await loadVocabulary(pathToVocabulary);
    const tokenizer = new Tokenizer(vocabulary);
    return tokenizer;
}

/**
 * Load a vocabulary for the Tokenizer.
 *
 * @param pathToVocabulary Defaults to the path to the 8k vocabulary used by the
 * UniversalSentenceEncoder.
 */
async function loadVocabulary(pathToVocabulary) {
    if (pathToVocabulary.startsWith('file://')) {
        pathToVocabulary = pathToVocabulary.replace('file://', '');
        const vocabulary = JSON.parse(fs.readFileSync(pathToVocabulary));
        return vocabulary;
    }

    const vocabulary = await tf.util.fetch(pathToVocabulary);
    return vocabulary.json();
}

module.exports = {
    Tokenizer,
    loadTokenizer,
    loadVocabulary
};