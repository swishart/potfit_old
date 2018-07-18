import glob
import os
import pytest
import random
import string
import subprocess

class Potfit:
    def __init__(self, location, model, interaction, options = []):
        self.cwd = os.path.dirname(location)
        self.model = model
        self.interaction = interaction
        self.options = options
        self.filenames = []
        self._build()

    def reset(self):
        self.filenames = []
        self.stdout = str()
        self.stderr = str()
        self.endpot = str()
        self.force = str()
        self.energy = str()
        self.stress = str()
        self.error = str()
        self.punish = str()
        self.rho_loc = str()
        self.returncode = 0

    def create_file(self, filename, **kwargs):
        name = os.path.join(self.cwd, filename)
        self.filenames.append(name)
        f = open(name, 'w')
        if 'permission' in kwargs:
            os.chmod(name, kwargs['permission'])
        return f

    def create_param_file(self, **kwargs):
        f = self.create_file('param_file')
        if 'ntypes' in kwargs:
            f.write('ntypes {}\n'.format(int(kwargs['ntypes'])))
            del(kwargs['ntypes'])
        else:
            f.write('ntypes 1\n')
        if 'startpot' in kwargs:
            f.write('startpot {}\n'.format(kwargs['startpot']))
        else:
            f.write('startpot startpot\n')
        if 'endpot' in kwargs:
            f.write('endpot {}\n'.format(kwargs['endpot']))
        else:
            f.write('endpot endpot\n')
        if 'config' in kwargs:
            f.write('config {}\n'.format(kwargs['config']))
        else:
            f.write('config config\n')
        if 'tempfile' in kwargs:
            f.write('tempfile {}\n'.format(kwargs['tempfile']))
        else:
            f.write('tempfile tempfile\n')
        if 'output_prefix' in kwargs:
            f.write('output_prefix {}\n'.format(kwargs['output_prefix']))
        else:
            f.write('output_prefix output\n')
        for item in kwargs:
            f.write('{} {}\n'.format(item, kwargs[item]))
        f.close()
        return

    def create_potential_file(self, input):
        f = self.create_file('startpot')
        f.write(input)
        f.close()

    def call_makeapot(self, filename, args):
        os.environ['PATH'] = '{}:'.format(os.path.abspath('../util')) + os.environ['PATH']
        p = subprocess.Popen(['makeapot', '-o', filename] + args.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd)
        p.wait()
        if p.returncode:
            pytest.fail('error calling "makeapot {}"'.format(args))
        self.filenames.append(filename)

    def create_config_file(self, **kwargs):
        f = self.create_file('config')
        basic_size = 3.0
        repeat_cell = 2
        ntypes = 1
        useforce = 1
        energy = -1.0
        if 'repeat_cell' in kwargs:
            repeat_cell = int(kwargs['repeat_cell'])
        if 'seed' in kwargs:
            random.seed(kwargs['seed'])
        if 'ntypes' in kwargs:
            ntypes = int(kwargs['ntypes'])
        if 'useforce' in kwargs:
            useforce = int(kwargs['useforce'])
        if 'energy' in kwargs:
            energy = float(kwargs['energy'])
        f.write('#N {} {}\n'.format(2 * repeat_cell * repeat_cell * repeat_cell, useforce))
        f.write('#C {}\n'.format(' '.join([str(x) for x in range(ntypes)])))
        f.write('## force file generated by potfit.py module\n')
        f.write('#X {} 0.0 0.0\n'.format(repeat_cell * basic_size))
        f.write('#Y 0.0 {} 0.0\n'.format(repeat_cell * basic_size))
        f.write('#Z 0.0 0.0 {}\n'.format(repeat_cell * basic_size))
        f.write('#W 1\n')
        f.write('#E {}\n'.format(energy))
        if 'stress' in kwargs and kwargs['stress'] == True:
            f.write('#S {}\n'.format(' '.join(str(random.uniform(-1,1)) for _ in range(6))))
        f.write('#F\n')
        for i in range(repeat_cell):
            for j in range(repeat_cell):
                for k in range(repeat_cell):
                    self._write_unit_cell(f, basic_size, ntypes, i, j, k)
        f.close()

    def run(self, param_file='param_file', args=[]):
        asan_filename = 'asan_{}'.format(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
        os.environ['ASAN_OPTIONS'] = 'log_path={},exitcode=99,strip_path_prefix={}'.format(asan_filename,os.path.abspath('..') + '/build/../')
        cmd = [os.path.join(os.path.abspath('../bin'), self.binary_name)]
        if len(args):
            cmd.extend(args)
        if param_file != None:
            cmd.append(param_file)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd)
        p.wait()
        self.stdout = p.stdout.read().decode('ascii')
        self.stderr = p.stderr.read().decode('ascii')
        try:
            f = open(os.path.join(self.cwd, param_file), 'r')
            for line in f:
                for token in ['startpot', 'endpot']:
                  if token in line:
                      filename = os.path.join(self.cwd, line.split()[1])
                      if os.path.isfile(filename):
                          if not filename in self.filenames:
                              self.filenames.append(filename)
                          g = open(filename)
                          setattr(self, token, g.read())
                          g.close()
                if 'output_prefix' in line:
                    filenames = [x for x in glob.iglob(os.path.join(self.cwd, line.split()[1] + '*')) if os.path.isfile(x)]
                    for fname in filenames:
                        self.filenames.append(fname)
                        g = open(fname)
                        setattr(self, os.path.basename(fname).split('.')[1], g.read())
                        g.close()
            f.close()
        except:
            pass
        self.returncode = p.returncode
        self._check_asan_fail(asan_filename)

    def has_no_error(self):
        if self.returncode != 0:
            return False
        return not '[ERROR]' in self.stderr

    def has_no_warning(self):
        return not '[WARNING]' in self.stderr

    def has_error(self):
        if self.returncode == 0:
            return False
        return '[ERROR]' in self.stderr

    def clear(self):
        for item in list(set(self.filenames)):
            if os.path.isfile(item):
                os.remove(item)

    def _write_unit_cell(self, f, basic_size, ntypes, i, j, k):
        f.write('{} {} {} {} {} {} {}\n'.format(random.randint(0, ntypes - 1), i * basic_size, j * basic_size, k * basic_size, -random.random(), -random.random(), -random.random()))
        f.write('{} {} {} {} {} {} {}\n'.format(random.randint(0, ntypes - 1), (i + 0.5) * basic_size + random.uniform(-0.2,0.2), (j + 0.5) * basic_size + random.uniform(-0.2,0.2), (k + 0.5) * basic_size + random.uniform(-0.2,0.2), -random.random(), -random.random(), -random.random()))

    def _build(self):
        p = subprocess.Popen(['./waf', 'distclean'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..')
        p.wait()
        if p.returncode:
            pytest.fail('error calling "waf distclean"')
        p = subprocess.Popen(self._get_conf_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..')
        p.wait()
        if p.returncode:
            print(p.stderr.read().decode('ascii'))
            pytest.fail('error calling "waf configure"')
        p = subprocess.Popen(['./waf', 'build'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..')
        p.wait()
        if p.returncode:
            print(p.stderr.read().decode('ascii'))
            pytest.fail('error calling "waf build"')
        out = p.stdout.read().decode('ascii').split('\n')
        self.binary_name = [x for x in out if 'Linking' in x][0].split(' ')[2].split('/')[-1]
        if not len(self.binary_name):
            pytest.fail('error reading binary name')

    def _get_conf_cmd(self):
        cmd = ['./waf', 'configure', '-c', 'no', '-m', self.model, '-i', self.interaction]
        for opt in self.options:
            cmd.append('--enable-{}'.format(opt))
        cmd.append('--asan')
        cmd.append('--debug')
        cmd.append('--check-c-compiler=clang')
        return cmd

    def _check_asan_fail(self, filename):
        if self.returncode != 99:
            return
        filenames = [x[len(os.path.abspath('..'))+7:] for x in glob.iglob(os.path.join(self.cwd,'{}.*'.format(filename))) if os.path.isfile(x)]
        pytest.fail('address sanitizer detected an error, please check {}'.format('\n'.join(filenames)))