#!/usr/bin/env python3
"""Propagate IV elasticities from advanced_elasticity_results.json into simulator_config.json"""
import json
import numpy as np

with open('dunnhumby/outputs/advanced_elasticity_results.json') as f:
    adv = json.load(f)

with open('dunnhumby/outputs/simulator_config.json') as f:
    config = json.load(f)

hybrid = adv['hybrid_elasticities']
products = config['products']
updated = 0

for i, p in enumerate(products):
    orig_pid = str(p.get('original_product_id', p.get('product_id', i)))
    if orig_pid in hybrid:
        old_e = p['self_elasticity']
        new_e = hybrid[orig_pid]
        p['self_elasticity'] = new_e
        if abs(old_e - new_e) > 0.01:
            print(f"  {p['category'][:25]:<25} OLS={old_e:.2f} -> IV={new_e:.2f}")
            updated += 1

config['elasticity_source'] = 'hybrid_iv_ols'
elast_list = [p['self_elasticity'] for p in products]
config['elasticities'] = elast_list

with open('dunnhumby/outputs/simulator_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nUpdated {updated} product elasticities")
print(f"Mean elasticity: {np.mean(elast_list):.2f}")
print(f"Range: [{min(elast_list):.2f}, {max(elast_list):.2f}]")
