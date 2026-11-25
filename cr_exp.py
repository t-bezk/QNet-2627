import numpy as np
import matplotlib.pyplot as plt
import monte_carlo as mc
import order_parameter as op


if __name__ == '__main__':

    BG = 13   # Number of generations
    P_TH = 0.5 * (np.sqrt(5) - 1)
    DELTAS = np.logspace(-1.7, -1.3, 8)    # p - p_th window
    MONTE_C = False     ## Monte-carlo (only use for small NO)

    ## Order Parameter from Star Mesh
    P_B = np.array([op.get_node_param(BG, P_TH + d)[0] for d in DELTAS])


    ## Order Parameter for n-2 generation checks to see if beta settles
    #P_B_CHK = np.array([get_node_param(BG - 2, P_TH + d)[0] for d in DELTAS])
    #print(f"drift between g={BG} and g={BG-2}: {np.max(np.abs(P_B / P_B_CHK - 1)):.1e}"
    #      "  (should be < 1e-3, otherwise move the window to larger delta)")

    x, y = np.log(DELTAS), np.log(P_B)
    slopes = np.diff(y) / np.diff(x)
    (p0, p1), cov = np.polyfit(x, y, 1, cov=True)
    beta_mean, beta_std = slopes.mean(), slopes.std(ddof=1)
    print(f"beta (mean of local slopes) = {beta_mean:.4f} +- {beta_std:.4f}")
    print(f"beta (straight-line fit)    = {p0:.4f} +- {np.sqrt(cov[0, 0]):.4f}")


## log-log plot with fit
    plt.figure()

    if MONTE_C: ## Monte-carlo (only use for small NO)
        fl = mc.uv_flower(2,2,BG)
        for p in DELTAS:
            P_B_MC = np.array([mc.mc_order_parameter(fl, P_TH + d,trials=100)[0] for d in DELTAS])
        plt.loglog(DELTAS,P_B_MC, 'o', label=f'OP. monte carlo n={BG}')

    plt.loglog(DELTAS, P_B, 'o', label=f'P_inf, n={BG}')
    plt.loglog(DELTAS, np.exp(p1) * DELTAS ** p0, '-', label=f'fit: beta = {p0:.4f}')

    plt.xlabel('p - p_th')
    plt.ylabel('P_inf')

    plt.legend()
    plt.grid(True, which='both', alpha=0.3)

    plt.savefig('figures/beta_loglog.png')
    plt.show()


    ## local slope with mean and fluctuation band
    plt.figure()

    plt.semilogx(np.sqrt(DELTAS[1:] * DELTAS[:-1]), slopes, 's-', label='local slope')
    plt.axhline(beta_mean, color='k', ls='--',
                label=f'mean = {beta_mean:.4f} +- {beta_std:.4f}')
    plt.axhspan(beta_mean - beta_std, beta_mean + beta_std, color='k', alpha=0.1)
    plt.xlabel('p - p_th')
    plt.ylabel('local beta')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('figures/beta_local.png')
    plt.show()
