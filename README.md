### Python Logistic Regression

This python program implements a logistic regression algorithm that identifies data between 2 handwritten digits from the [handwritten digits dataset](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

#### Activation Function
Activation function used is the Sigmoid.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>h</mi>
    <mi>&#x03B8;<!-- θ --></mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow class="MJX-TeXAtom-ORD">
      <mn>1</mn>
    </mrow>
    <mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mn>1</mn>
      </mrow>
      <mo>+</mo>
      <msup>
        <mi>e</mi>
        <mo>&#x2212;<!-- − --></mo>
      </msup>
      <msup>
        <mi>&#x03B8;<!-- θ --></mi>
        <mi>T</mi>
      </msup>
      <mi>x</mi>
    </mrow>
  </mfrac>
</math>

### Loss Function
Cross Entropy Loss Function is being used.

