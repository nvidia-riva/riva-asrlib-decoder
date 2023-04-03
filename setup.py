# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import shutil

import setuptools

from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel

def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs['language'] = 'c++'
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):

    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        riva_asrlib_decoder_dir = os.path.dirname(os.path.abspath(__file__))

        cmake_args = os.environ.get('RIVA_ASRLIB_DECODER_CMAKE_ARGS', '')
        make_args = os.environ.get('RIVA_ASRLIB_DECODER_MAKE_ARGS', '-j 16')
        
        system_make_args = os.environ.get('MAKEFLAGS', '')

        cmake_args += " -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=ON "

        if (
            make_args == ""
            and system_make_args == ""
            and os.environ.get("RIVA_ASRLIB_DECODER_IS_GITHUB_ACTIONS", None) is None
        ):
            print("For fast compilation, run:")
            print('export RIVA_ASRLIB_DECODER_MAKE_ARGS="-j"; python setup.py install')

        # if 'PYTHON_EXECUTABLE' not in cmake_args:
        #     print(f'Setting PYTHON_EXECUTABLE to {sys.executable}')
        #     cmake_args += f' -DPYTHON_EXECUTABLE={sys.executable}'

        # TODO: Remove this dead code
        openfst_binaries = [f"fst{operation}" for operation in "arcsort closure compile concat connect convert determinize disambiguate encode epsnormalize equal equivalent invert isomorphic map minimize project prune push randgen relabel replace reverse reweight synchronize topsort union".split(" ")]
        fst_binaries = "arpa2fst arpa-to-const-arpa fstdeterminizestar fstrmsymbols fstisstochastic fstminimizeencoded fstmakecontextfst fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops fstrmepslocal fstcomposecontext fsttablecompose fstrand fstdeterminizelog fstphicompose fstcopy fstpushspecial fsts-to-transcripts fsts-project fsts-union fsts-concat transcripts-to-fsts".split(" ")
        fst_binaries.extend(openfst_binaries)
        # ERROR to fix: "fstcompose" is used in mkgraph_ctc.sh...
        fst_binaries = ["arpa2fst", "fsttablecompose", "fstdeterminizestar", "fstminimizeencoded", "fstarcsort", "fstcompile", "fstaddselfloops", "transcripts-to-fsts", "fstconvert", "fstisstochastic", "fstcompose", "fstrmepslocal"]
        build_cmd = f'''
        cd {self.build_temp}
        
        cmake -GNinja {cmake_args} {riva_asrlib_decoder_dir}

        cmake --build . -- {make_args} python_decoder {' '.join(fst_binaries)} offline-cuda-decode-binary
        '''
        print(f'build command is:\n{build_cmd}')

        ret = os.system(build_cmd)
        if ret != 0:
            raise Exception('Failed to build riva_asrlib_decoder')

        lib_so = glob.glob(f'{self.build_temp}/*python_decoder*.so')
        for so in lib_so:
            print(f'Copying {so} to {self.build_lib}/riva/asrlib/decoder')
            os.makedirs(f'{self.build_lib}/riva/asrlib/decoder', exist_ok=True)
            shutil.copy(f'{so}', f'{self.build_lib}/riva/asrlib/decoder')

        dir_path = os.path.dirname(os.path.realpath(__file__))

        for fst_binary in fst_binaries:
            os.makedirs(f'{self.build_lib}/riva/asrlib/decoder/scripts/prepare_TLG_fst/bin', exist_ok=True)
            # This copy does not work when installing in editable mode, hmm...
            shutil.copy(f"{self.build_temp}/{fst_binary}", f'{self.build_lib}/riva/asrlib/decoder/scripts/prepare_TLG_fst/bin')
            os.makedirs(f'{dir_path}/src/riva/asrlib/decoder/scripts/prepare_TLG_fst/bin', exist_ok=True)
            shutil.copy(f'{self.build_lib}/riva/asrlib/decoder/scripts/prepare_TLG_fst/bin/{fst_binary}',
                        f'{dir_path}/src/riva/asrlib/decoder/scripts/prepare_TLG_fst/bin')

setuptools.setup(
    python_requires='>=3.7',
    name='riva-asrlib-decoder',
    version='0.3.1',
    author='NVIDIA',
    author_email='dgalvez@nvidia.com',
    keywords='ASR, CUDA, WFST, Decoder',
    description="Implementation of https://arxiv.org/abs/1910.10032",
    url='https://github.com/nvidia-riva/riva-asrlib-decoder',
    package_dir={
        '': 'src',
        'riva.asrlib.decoder': 'src/riva/asrlib/decoder',
    },
    packages=['riva.asrlib.decoder'],
    # package_data={'riva.asrlib.decoder': ['todo.txt']},
    package_data={
        'riva.asrlib.decoder': [
            "todo.txt",
            "scripts/prepare_TLG_fst/local/*.py",
                                "scripts/prepare_TLG_fst/local/*.sh",
                                "scripts/prepare_TLG_fst/utils/*.pl",
                                # "scripts/prepare_TLG_fst/bin/*",
                                "scripts/prepare_TLG_fst/*.py",
                                "scripts/prepare_TLG_fst/*.sh"]
    },
    ext_modules=[cmake_extension('riva.asrlib.decoder.python_decoder')],
    cmdclass={
        'build_ext': BuildExtension,
        'bdist_wheel': bdist_wheel
    },
    # scripts=(glob.glob("scripts/prepare_TLG_fst/**/*.py") +
    #          glob.glob("scripts/prepare_TLG_fst/**/*.sh")),
    include_package_data=True,
    install_requires=["sentencepiece",],
    extras_require={
        'testing': [
            "pytest",
            "kaldi-io",
            "more-itertools",
            "nemo_toolkit[asr]",
            "torchaudio",
        ]
    },
    zip_safe=False,
)
