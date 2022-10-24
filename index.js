// JavaScript:
const tf = require('@tensorflow/tfjs-core');
const tfconv = require('@tensorflow/tfjs-converter');

const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder';

const loadTokenizer = require('./tokenizer').loadTokenizer;
const loadVocabulary = require('./tokenizer').loadVocabulary;
const Tokenizer = require('./tokenizer').Tokenizer;
const loadQnA = require('./use_qna').loadQnA;

const version = require('./version').version;

async function load(config) {
    const use = new UniversalSentenceEncoder();
    await use.load(config);
    return use;
}

class UniversalSentenceEncoder {
    async loadModel(modelUrl) {
        return modelUrl
            ? tfconv.loadGraphModel(modelUrl)
            : tfconv.loadGraphModel(
                'https://tfhub.dev/tensorflow/tfjs-model/universal-sentence-encoder-lite/1/default/1',
                { fromTFHub: true }
            );
    }

    async load(config = {}) {
        const [model, vocabulary] = await Promise.all([
            this.loadModel(config.modelUrl),
            loadVocabulary(config.vocabUrl || `${BASE_PATH}/vocab.json`)
        ]);

        this.model = model;
        this.tokenizer = new Tokenizer(vocabulary);
    }

    /**
     *
     * Returns a 2D Tensor of shape [input.length, 512] that contains the
     * Universal Sentence Encoder embeddings for each input.
     *
     * @param inputs A string or an array of strings to embed.
     */
    async embed(inputs) {
        if (typeof inputs === 'string') {
            inputs = [inputs];
        }

        const encodings = inputs.map(d => this.tokenizer.encode(d));

        const indicesArr =
            encodings.map((arr, i) => arr.map((d, index) => [i, index]));

        let flattenedIndicesArr = [];
        for (let i = 0; i < indicesArr.length; i++) {
            flattenedIndicesArr = flattenedIndicesArr.concat(indicesArr[i]);
        }

        const indices = tf.tensor2d(
            flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
        const values = tf.tensor1d(tf.util.flatten(encodings), 'int32');

        const modelInputs = { indices, values };

        const embeddings = await this.model.executeAsync(modelInputs);
        indices.dispose();
        values.dispose();

        return embeddings;
    }
}

module.exports = {
    load,
    UniversalSentenceEncoder,
    Tokenizer,
    loadTokenizer,
    loadQnA,
    version
};