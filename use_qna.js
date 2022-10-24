const tfconv = require('@tensorflow/tfjs-converter');
const tf = require('@tensorflow/tfjs-core');

const loadVocabulary = require('./tokenizer').loadVocabulary;
const Tokenizer = require('./tokenizer').Tokenizer;

const version = require('./version').version;

const BASE_PATH =
    'https://tfhub.dev/google/tfjs-model/universal-sentence-encoder-qa-ondevice/1';
// Index in the vocab file that needs to be skipped.
const SKIP_VALUES = [0, 1, 2];
// Offset value for skipped vocab index.
const OFFSET = 3;
// Input tensor size limit.
const INPUT_LIMIT = 192;
// Model node name for query.
const QUERY_NODE_NAME = 'input_inp_text';
// Model node name for query.
const RESPONSE_CONTEXT_NODE_NAME = 'input_res_context';
// Model node name for response.
const RESPONSE_NODE_NAME = 'input_res_text';
// Model node name for response result.
const RESPONSE_RESULT_NODE_NAME = 'Final/EncodeResult/mul';
// Model node name for query result.
const QUERY_RESULT_NODE_NAME = 'Final/EncodeQuery/mul';
// Reserved symbol count for tokenizer.
const RESERVED_SYMBOLS_COUNT = 3;
// Value for token padding
const TOKEN_PADDING = 2;
// Start value for each token
const TOKEN_START_VALUE = 1;

async function loadQnA() {
    const use = new UniversalSentenceEncoderQnA();
    await use.load();
    return use;
}

class UniversalSentenceEncoderQnA {
    async loadModel() {
        return tfconv.loadGraphModel(BASE_PATH, { fromTFHub: true });
    }

    async load() {
        const [model, vocabulary] = await Promise.all([
            this.loadModel(),
            loadVocabulary(`${BASE_PATH}/vocab.json?tfjs-format=file`)
        ]);

        this.model = model;
        this.tokenizer = new Tokenizer(vocabulary, RESERVED_SYMBOLS_COUNT);
    }

    /**
     *
     * Returns a map of queryEmbedding and responseEmbedding
     *
     * @param input the ModelInput that contains queries and answers.
     */
    embed(input) {
        const embeddings = tf.tidy(() => {
            const queryEncoding = this.tokenizeStrings(input.queries, INPUT_LIMIT);
            const responseEncoding =
                this.tokenizeStrings(input.responses, INPUT_LIMIT);
            if (input.contexts != null) {
                if (input.contexts.length !== input.responses.length) {
                    throw new Error(
                        'The length of response strings ' +
                        'and context strings need to match.');
                }
            }
            const contexts = input.contexts || [];
            if (input.contexts == null) {
                contexts.length = input.responses.length;
                contexts.fill('');
            }
            const contextEncoding = this.tokenizeStrings(contexts, INPUT_LIMIT);
            const modelInputs = {};
            modelInputs[QUERY_NODE_NAME] = queryEncoding;
            modelInputs[RESPONSE_NODE_NAME] = responseEncoding;
            modelInputs[RESPONSE_CONTEXT_NODE_NAME] = contextEncoding;

            return this.model.execute(
                modelInputs, [QUERY_RESULT_NODE_NAME, RESPONSE_RESULT_NODE_NAME]);
        });
        const queryEmbedding = embeddings[0];
        const responseEmbedding = embeddings[1];

        return { queryEmbedding, responseEmbedding };
    }

    tokenizeStrings(strs, limit) {
        const tokens =
            strs.map(s => this.shiftTokens(this.tokenizer.encode(s), INPUT_LIMIT));
        return tf.tensor2d(tokens, [strs.length, INPUT_LIMIT], 'int32');
    }

    shiftTokens(tokens, limit) {
        tokens.unshift(TOKEN_START_VALUE);
        for (let index = 0; index < limit; index++) {
            if (index >= tokens.length) {
                tokens[index] = TOKEN_PADDING;
            } else if (!SKIP_VALUES.includes(tokens[index])) {
                tokens[index] += OFFSET;
            }
        }
        return tokens.slice(0, limit);
    }
}

module.exports = {
    version,
    loadQnA,
    UniversalSentenceEncoderQnA
};