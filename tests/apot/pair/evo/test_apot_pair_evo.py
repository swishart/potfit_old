import pytest

def test_apot_pair_evo_threshold_empty(potfit):
    potfit.create_param_file(evo_threshold='')
    potfit.call_makeapot('startpot', '-n 1 -i pair -f eopp_sc')
<<<<<<< HEAD
    potfit.create_config_file(repeat_cell=3, seed=42)
=======
    potfit.create_config_file()
>>>>>>> 9954275972b5a639aeab1e6816558889f228773f
    potfit.run()
    assert potfit.has_error()
    assert 'Missing value in parameter file' in potfit.stderr
    assert 'evo_threshold is <undefined>' in potfit.stderr

def test_apot_pair_evo_threshold_invalid(potfit):
    potfit.create_param_file(evo_threshold='foo')
    potfit.call_makeapot('startpot', '-n 1 -i pair -f eopp_sc')
<<<<<<< HEAD
    potfit.create_config_file(repeat_cell=3, seed=42)
=======
    potfit.create_config_file()
>>>>>>> 9954275972b5a639aeab1e6816558889f228773f
    potfit.run()
    assert potfit.has_error()
    assert 'Illegal value in parameter file' in potfit.stderr
    assert 'evo_threshold is not a double' in potfit.stderr
    assert 'value = foo' in potfit.stderr

def test_apot_pair_evo_threshold_out_of_bounds(potfit):
    potfit.create_param_file(evo_threshold=-1)
    potfit.call_makeapot('startpot', '-n 1 -i pair -f eopp_sc')
<<<<<<< HEAD
    potfit.create_config_file(repeat_cell=3, seed=42)
=======
    potfit.create_config_file()
>>>>>>> 9954275972b5a639aeab1e6816558889f228773f
    potfit.run()
    assert potfit.has_error()
    assert 'Illegal value in parameter file' in potfit.stderr
    assert 'evo_threshold is out of bounds' in potfit.stderr
