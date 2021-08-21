﻿using NWaves.Filters.Base;
using NWaves.Filters.Fda;

namespace NWaves.Filters.Elliptic
{
    public class HighPassFilter : ZiFilter
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="freq"></param>
        /// <param name="order"></param>
        /// <param name="ripplePass"></param>
        /// <param name="rippleStop"></param>
        public HighPassFilter(double freq, int order, double ripplePass = 1, double rippleStop = 20) :
            base(MakeTf(freq, order, ripplePass, rippleStop))
        {
        }

        /// <summary>
        /// TF generator
        /// </summary>
        /// <param name="freq"></param>
        /// <param name="order"></param>
        /// <param name="ripplePass"></param>
        /// <param name="rippleStop"></param>
        /// <returns></returns>
        private static TransferFunction MakeTf(double freq, int order, double ripplePass = 1, double rippleStop = 20)
        {
            return DesignFilter.IirHpTf(freq,
                                        PrototypeElliptic.Poles(order, ripplePass, rippleStop),
                                        PrototypeElliptic.Zeros(order, ripplePass, rippleStop));
        }

        /// <summary>
        /// Change filter coeffs online
        /// </summary>
        /// <param name="freq"></param>
        /// <param name="ripplePass"></param>
        /// <param name="rippleStop"></param>
        public void Change(double freq, double ripplePass = 1, double rippleStop = 20)
        {
            Change(MakeTf(freq, _a.Length - 1, ripplePass, rippleStop));
        }
    }
}
