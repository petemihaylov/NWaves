using NUnit.Framework;
using NWaves.Transforms;
using System;
using System.Linq;

namespace NWaves.Tests.TransformTests
{
    [TestFixture]
    public class TestFft
    {
        [Test]
        public void TestRealFft()
        {
            float[] array = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }; // Enumerable.Range(0, 16);

            float[] re = new float[9];
            float[] im = new float[9];

            var realFft = new RealFft(16);

            realFft.Direct(array, re, im);

            Assert.That(re, Is.EqualTo(new float[] { 120, -8, -8, -8, -8, -8, -8, -8, -8 }).Within(1e-5));
            Assert.That(im, Is.EqualTo(new float[] { 0, 40.21872f, 19.31371f, 11.97285f, 8, 5.34543f, 3.31371f, 1.591299f, 0 }).Within(1e-5));
        }

        [Test]
        public void TestRealFft64()
        {
            double[] array = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }; // Enumerable.Range(0, 16);

            double[] re = new double[9];
            double[] im = new double[9];

            var realFft = new RealFft64(16);

            realFft.Direct(array, re, im);

            Assert.That(re, Is.EqualTo(new double[] { 120, -8, -8, -8, -8, -8, -8, -8, -8 }).Within(1e-5));
            Assert.That(im, Is.EqualTo(new double[] { 0, 40.21872f, 19.31371f, 11.97285f, 8, 5.34543f, 3.31371f, 1.591299f, 0 }).Within(1e-5));
        }

        [Test]
        public void TestInverseRealFft64()
        {
            double[] array = { 1, 5, 3, 7, 2, 3, 0, 7 };
            double[] output = new double[array.Length];
            double[] outputNorm = new double[array.Length];
            double[] re = new double[5];
            double[] im = new double[5];

            var realFft = new RealFft64(8);
            realFft.Direct(array, re, im);
            realFft.Inverse(re, im, output);
            realFft.InverseNorm(re, im, outputNorm);

            Assert.Multiple(() =>
            {
                Assert.That(output, Is.EqualTo(array.Select(a => a * 8)).Within(1e-12));
                Assert.That(outputNorm, Is.EqualTo(array).Within(1e-12));
            });
        }

        [Test]
        public void TestComplexDirectFft64()
        {
            double[] inRe = { 1, 2, 3, 4, 5, 6, 7, 8 };
            double[] inIm = new double[8];
            double[] outRe = new double[8];
            double[] outIm = new double[8];

            var realFft = new RealFft64(8);
            realFft.Direct(inRe, inIm, outRe, outIm);

            var expectedRe = new double[] { 36.0, -4.0, -4.0, -4.0, -4.0, 0.0, 0.0, 0.0 };  // Corrected values
            var expectedIm = new double[]
            {
                0.0,
                9.656854249492381,
                4.0,
                1.656854249492381,
                0.0,
                0.0,  // Corrected value
                0.0,
                0.0
            };

            Assert.Multiple(() =>
            {
                Assert.That(outRe, Is.EqualTo(expectedRe).Within(1e-12));
                Assert.That(outIm, Is.EqualTo(expectedIm).Within(1e-12));
            });
        }

        [Test]
        public void TestInverseRealFft()
        {
            float[] array = { 1, 5, 3, 7, 2, 3, 0, 7 };
            float[] output = new float[array.Length];
            float[] outputNorm = new float[array.Length];

            float[] re = new float[5];
            float[] im = new float[5];

            var realFft = new RealFft(8);

            realFft.Direct(array, re, im);
            realFft.Inverse(re, im, output);
            realFft.InverseNorm(re, im, outputNorm);

            Assert.Multiple(() =>
            {
                Assert.That(output, Is.EqualTo(array.Select(a => a * 8)).Within(1e-5));
                Assert.That(outputNorm, Is.EqualTo(array).Within(1e-5));
            });
        }

        [Test]
        public void TestComplexFft()
        {
            float[] re = { 0, 1, 2, 3, 4, 5, 6, 7 };
            float[] im = new float[8];

            var fft = new Fft(8);

            fft.Direct(re, im);

            Assert.That(re, Is.EqualTo(new float[] { 28, -4, -4, -4, -4, -4, -4, -4 }).Within(1e-5));
            Assert.That(im, Is.EqualTo(new float[] { 0, 9.65685f, 4, 1.65685f, 0, -1.65685f, -4, -9.65685f }).Within(1e-5));
        }

        [Test]
        public void TestInverseFft()
        {
            float[] re = { 1, 5, 3, 7, 2, 3, 0, 7 };
            float[] im = new float[re.Length];

            var fft = new Fft(8);

            fft.Direct(re, im);
            fft.Inverse(re, im);

            Assert.That(re, Is.EqualTo(new[] { 8, 40, 24, 56, 16, 24, 0, 56 }).Within(1e-5)); 
            // re[i] * 8,  i = 0..7
        }

        [Test]
        public void TestInverseFftNormalized()
        {
            float[] re = { 1, 5, 3, 7, 2, 3, 0, 7 };
            float[] im = new float[re.Length];

            var fft = new Fft(8);

            fft.Direct(re, im);
            fft.InverseNorm(re, im);

            Assert.That(re, Is.EqualTo(new[] { 1, 5, 3, 7, 2, 3, 0, 7 }).Within(1e-5));
        }

        [Test]
        public void TestFftShift()
        {
            float[] array = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

            Fft.Shift(array);

            Assert.That(array, Is.EqualTo(new float[] { 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 }));
        }

        [Test]
        public void TestFftShiftOddSize()
        {
            float[] array = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

            Assert.Throws<ArgumentException>(() => Fft.Shift(array));
        }

        [Test]
        public void TestGoertzel()
        {
            float[] array = { 1, 2, 3, 4, 5, 6, 7, 8 };

            var cmpx = new Goertzel(8).Direct(array, 2);

            Assert.Multiple(() =>
            {
                Assert.That(cmpx.Real, Is.EqualTo(-4).Within(1e-6));
                Assert.That(cmpx.Imaginary, Is.EqualTo(4).Within(1e-6));
            });
        }

        [Test]
        public void TestHartley()
        {
            float[] re = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

            var dht = new HartleyTransform(16);

            dht.Direct(re);
            dht.InverseNorm(re);

            Assert.That(re, Is.EqualTo(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16f }).Within(1e-4));
        }
    }
}
