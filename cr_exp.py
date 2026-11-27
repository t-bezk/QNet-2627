import numpy as np
import matplotlib.pyplot as plt
import monte_carlo as mc
import order_parameter as op


P_TH    =   0.5 * (np.sqrt(5) - 1)      ## Analytic critical Value (classical sys.)

C_TH    =   0.431246

def high_gen_plt(gen):
    
    delta_p  =   np.logspace(-1.7, -1.3, 8)  ## p - p_th window

    ## Order Parameter from Star Mesh
    p_b = np.array([op.get_node_param(gen, P_TH + d)[0] for d in delta_p])

    ## Estimate beta from star mesh
    x, y = np.log(delta_p), np.log(p_b)
    slopes = np.diff(y) / np.diff(x)
    (p0, p1), cov = np.polyfit(x, y, 1, cov=True)   ## Fit to a straight line
    beta_mu, beta_std = slopes.mean(), slopes.std(ddof=1)
    print(f"beta (local mean) = {beta_mu:.4f} +- {beta_std:.4f}")
    print(f"beta (straight line fit)    = {p0:.4f} +- {np.sqrt(cov[0, 0]):.4f}")

    ## log-log plot with fit
    plt.figure()

    plt.loglog(delta_p, p_b, 'o', label=f'P_inf, n={gen}')
    plt.loglog(delta_p, np.exp(p1) * delta_p ** p0, '-', label=f'fit: beta = {p0:.4f}')

    plt.xlabel('p - p_th')
    plt.ylabel('P_inf')

    plt.legend()
    plt.grid()

    plt.savefig(f'figures/beta_loglog_n={gen}.png')
    plt.close()
    
    plt.figure()
    p = np.linspace(0.4,0.8,25)
    ## Order Parameter from Star Mesh
    p_b = np.array([op.get_node_param(gen, _p)[0] for _p in p])
    p_sc = np.array([op.get_node_param(gen, _p)[1] for _p in p])
    plt.plot(p,p_b,label='p_inf')
    plt.plot(p,p_sc,label='p_sc')
    plt.legend()
    plt.grid()
    plt.xlabel('p')
    plt.ylabel('Parameter')
    plt.savefig(f'figures/pinfsc_n={gen}.png')
    plt.show()



def get_beta_from_sgen(gen, monte_carlo=False, gen_graphs=False, show_graphs=False):

    delta_p  =   np.logspace(-1.7, -1.3, 8)  ## p - p_th window

    ## Order Parameter from Star Mesh
    p_b = np.array([op.get_node_param(gen, P_TH + d)[0] for d in delta_p])

    ## Estimate beta from star mesh
    x, y = np.log(delta_p), np.log(p_b)
    slopes = np.diff(y) / np.diff(x)
    (p0, p1), cov = np.polyfit(x, y, 1, cov=True)   ## Fit to a straight line
    beta_mu, beta_std = slopes.mean(), slopes.std(ddof=1)
    #print(f"beta (local mean) = {beta_mu:.4f} +- {beta_std:.4f}")
    #print(f"beta (straight line fit)    = {p0:.4f} +- {np.sqrt(cov[0, 0]):.4f}")

    ## log-log plot with fit
    plt.figure()

    if monte_carlo: ## Monte-carlo (only use for small NO)
        fl = mc.uv_flower(2,2,gen)
        p_b_mc = np.array([mc.mc_order_parameter(fl, P_TH + d,trials=100)[0] for d in delta_p])
        plt.loglog(delta_p,p_b_mc, 'o', label=f'OP. monte carlo n={gen}')

        xmc, ymc = np.log(delta_p), np.log(p_b_mc)
        slopesmc = np.diff(ymc) / np.diff(xmc)
        (q0, _), covmc = np.polyfit(xmc, ymc, 1, cov=True)   ## Fit to a straight line
        beta_mu_mc, beta_std_mc = slopesmc.mean(), slopesmc.std(ddof=1)

    plt.loglog(delta_p, p_b, 'o', label=f'P_inf, n={gen}')
    plt.loglog(delta_p, np.exp(p1) * delta_p ** p0, '-', label=f'fit: beta = {p0:.4f}')

    plt.xlabel('p - p_th')
    plt.ylabel('P_inf')

    plt.legend()
    plt.grid()

    #plt.savefig('figures/beta_loglog.png')
    plt.close()



    ## local slope with mean and fluctuation band
    plt.figure()

    plt.semilogx(np.sqrt(delta_p[1:] * delta_p[:-1]), slopes, 's-', label='local slope')
    plt.axhline(beta_mu, color='k', ls='--', label=f'mean = {beta_mu:.4f} +- {beta_std:.4f}')
    plt.axhspan(beta_mu - beta_std, beta_mu + beta_std, color='k', alpha=0.1)

    plt.xlabel('p - p_th')
    plt.ylabel('local beta')

    plt.legend()
    plt.grid()

    #plt.savefig('figures/beta_local.png')
    plt.close()

    if monte_carlo:
        return beta_mu, beta_std, p0, np.sqrt(cov[0, 0]), beta_mu_mc, beta_std_mc, q0, np.sqrt(covmc[0, 0])

    return beta_mu, beta_std, p0, np.sqrt(cov[0, 0])

if __name__ == '__main__':

    #high_gen_plt(2)

    #w = []
    #print('Beta straight line approx. w/ Monte Carlo')
    #print('gen no | beta_mu | beta_std | beta_mc_mu | beta_mc_std')
    #for i in range(1,10):
    #    x = get_beta_from_sgen(i,monte_carlo=True)
    #    print(f'{i} | {x[2]} | {x[3]} | {x[6]} | {x[7]}')
    #    w.append([i,x[2],x[3],x[6],x[7]])
##
    #w = np.array(w).T
    #print(w[0])
    #plt.scatter(w[0],w[1],label='beta_mu_small')
    #plt.errorbar(w[0],w[1],yerr=w[2],ls='none',capsize=2.)
    #plt.scatter(w[0],w[3],label='beta_monte')
    #plt.errorbar(w[0],w[3],yerr=w[4],ls='none',capsize=2.)
    #plt.legend()
    #plt.grid()
    #plt.xlabel('generation number')
    #plt.ylabel('beta')
    #plt.savefig('figures/monte_sm_gen_mon_c_conc.png',dpi=500)
    #plt.show()
#
    w = []
    print('gen no | beta_mu | beta_std')
    for i in range(4,12):
        x = get_beta_from_sgen(i)
        print(f'{i} | {x[2]} | {x[3]}')
        w.append([i,x[2],x[3]])
##
    w = np.array(w).T
    plt.scatter(w[0],w[1],label='beta_mu_large')
    plt.errorbar(w[0],w[1],yerr=w[2],ls='none',capsize=2.)
    plt.legend()
    plt.grid()
    plt.xlabel('generation number')
    plt.ylabel('beta')
    plt.savefig('figures/monte_sm_gen_lrg_n.png',dpi=500)
    plt.show()
#