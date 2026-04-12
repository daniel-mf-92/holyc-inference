// softmax.HC
// Integer-only Q16.16 softmax for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

I64 FPQ16Exp(I64 x);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);

U0 FPQ16Softmax(I64 *logits, I64 *probs, I64 count)
{
    I64 i;
    I64 max_logit;
    I64 shifted;
    I64 exp_val;
    I64 exp_sum;
    I64 prob_sum;
    I64 diff;
    I64 max_idx;

    if (!logits || !probs || count <= 0)
        return;

    max_logit = logits[0];
    max_idx = 0;
    for (i = 1; i < count; i++)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
            max_idx = i;
        }
    }

    exp_sum = 0;
    for (i = 0; i < count; i++)
    {
        shifted = logits[i] - max_logit;
        exp_val = FPQ16Exp(shifted);
        probs[i] = exp_val;

        if (exp_sum > 0x7FFFFFFFFFFFFFFF - exp_val)
            exp_sum = 0x7FFFFFFFFFFFFFFF;
        else
            exp_sum += exp_val;
    }

    if (exp_sum <= 0)
    {
        exp_val = FPQ16Div(FP_Q16_ONE, FPQ16FromInt(count));
        for (i = 0; i < count; i++)
            probs[i] = exp_val;
        return;
    }

    prob_sum = 0;
    for (i = 0; i < count; i++)
    {
        probs[i] = FPQ16Div(probs[i], exp_sum);
        prob_sum += probs[i];
    }

    diff = FP_Q16_ONE - prob_sum;
    probs[max_idx] += diff;
}
