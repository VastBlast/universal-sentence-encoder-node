const path = require('path');
const { readFileSync } = require('fs');
const tf = require('@tensorflow/tfjs-core');
const tfconv = require('@tensorflow/tfjs-converter');

const BASE_PATH = 'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder';

const { loadTokenizer } = require('./tokenizer');
const { loadVocabulary } = require('./tokenizer');
const { Tokenizer } = require('./tokenizer');
const { loadQnA } = require('./use_qna');

const { version } = require('./version');

async function load(config) {
    const use = new UniversalSentenceEncoder();
    await use.load(config);
    return use;
}

const modelUrlDefault = 'file://' + path.relative('./', path.join(__dirname, './models/universal-sentence-encoder-lite_2/model.json')).replaceAll('\\', '/');
const vocabUrlDefault = 'file://' + path.relative('./', path.join(__dirname, './models/universal-sentence-encoder-lite_2/vocab.json')).replaceAll('\\', '/');

class UniversalSentenceEncoder {
    async loadModel(modelUrl) {
        return modelUrl
            ? tfconv.loadGraphModel(modelUrl, {
                fetchFunc: (path, requestInit) => {
                    //console.log('FETCH:', path);
                    if (path.startsWith('file://')) {
                        const filePath = path.replace('file://', '');
                        const data = readFileSync(filePath);
                        //console.log('DATA:', data);
                        return {
                            ok: true,
                            status: 200,
                            url: path,
                            arrayBuffer: () => Promise.resolve(data),
                            text: () => Promise.resolve(data.toString()),
                            json: () => Promise.resolve(JSON.parse(data.toString()))
                        }
                    }

                    return tf.util.fetch(path, requestInit);
                }
            })
            : tfconv.loadGraphModel(
                'https://tfhub.dev/tensorflow/tfjs-model/universal-sentence-encoder-lite/1/default/1',
                { fromTFHub: true }
            );
    }

    async load(config = {}) {
        if (!config.modelUrl) config.modelUrl = modelUrlDefault;
        if (!config.vocabUrl) config.vocabUrl = vocabUrlDefault;

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