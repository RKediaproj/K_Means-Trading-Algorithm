#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Compute Weights of Portfolio, Rebalance Monthly, Limit Leverage Ratio to 1

import quantopian.optimize as opt

MAX_GROSS_LEVERAGE = 1.0

def compute_target_weights(context, data):
    weights = {}
    
    if context.longs and context.shorts:
        long_weight = 0.5 / len(context.longs)
        short_weight = -0.5 / len(context.shorts)
    else:
        return weights
    
    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            weights[security] = 0

    for security in context.longs:
        weights[security] = long_weight

    for security in context.shorts:
        weights[security] = short_weight

    return weights


def rebalance(context, data):
    
    target_weights = compute_target_weights(context, data)

    if target_weights:
        order_optimal_portfolio(
            objective=opt.TargetWeights(target_weights),
            constraints=[opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)],) #opt.DollarNeutral()],)
        

def record_vars(context, data):
    
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1
    
    record(num_positions=len(context.portfolio.positions),
           leverage=context.account.leverage,
           long_count = longs,
           short_count = shorts)

